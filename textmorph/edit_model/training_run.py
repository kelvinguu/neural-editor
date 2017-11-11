import cPickle as pickle
import codecs
import random
from contextlib import contextmanager
from itertools import izip
from os import listdir
import os
from os.path import dirname, realpath, join

import numpy as np
import torch
import torch.optim as optim
from tensorboard_logger import tensorboard_logger
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from fabric.api import local

import gtd.io
from gtd.chrono import verboserate
from gtd.log import Metadata
from gtd.utils import random_seed, sample_if_large, bleu, Failure, Config, chunks
from gtd.ml.training_run import TrainingRunWorkspace, TrainingRuns
from gtd.ml.torch.training_run import TorchTrainingRun
from textmorph import data
from textmorph.edit_model.attention_decoder import AttentionDecoderCell
from gtd.ml.torch.simple_decoder_cell import SimpleDecoderCell
from textmorph.edit_model.edit_noiser import EditNoiser
from textmorph.edit_model.editor import Editor, EditExample
from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.torch.utils import similar_size_batches, try_gpu
from gtd.ml.vocab import SimpleEmbeddings, WordVocab

class EditTrainingRuns(TrainingRuns):
    def __init__(self, check_commit=True):
        data_dir = data.workspace.edit_runs
        src_dir = dirname(dirname(dirname(realpath(__file__))))  # root of the Git repo
        super(EditTrainingRuns, self).__init__(data_dir, src_dir, EditTrainingRun, check_commit=check_commit)

    @classmethod
    def default_format(cls, workspace):
        # get experiment name
        path = workspace.root
        name = os.path.split(path)[-1]

        # steps taken
        ckpt_nums = EditTrainingRun._checkpoint_numbers(workspace.checkpoints)
        steps = max(ckpt_nums) if ckpt_nums else 0

        # metadata
        meta = Metadata.from_file(workspace.metadata)
        bleu = meta.get('bleu_valid', None)
        loss = meta.get('loss_valid', None)
        dirty_repo = meta.get('dirty_repo', '?')

        # dataset
        config = Config.from_file(workspace.config)
        dataset = config.dataset.path

        return '{name:10} -- steps: {steps:<10}, loss: {loss:.2f}, dset: {dset:15}, bleu: {bleu:.2f} ' \
               'dirty_repo: {dirty_repo}'.format(
                name=name, dset=dataset, steps=steps, loss=loss, bleu=bleu, dirty_repo=dirty_repo)

    def summarize(self, fmt=None, verbose=False):
        if fmt is None:
            fmt = self.default_format

        for path in self.paths():
            ws = TrainingRunWorkspace(path)
            try:
                print fmt(ws)
            except BaseException:
                msg = 'Failed to render experiment: {}.'.format(ws.root)
                f = Failure.silent(msg)
                if verbose:
                    print msg
                    print f.traceback


class EditDataSplits(object):
    """
    Attributes:
        train (list[EditExample])
        valid (list[EditExample])
        test (list[EditExample])
        free (list[unicode]): a list of "free words" which are excluded from insertions and deletions
    """

    def __init__(self, data_dir, use_diff):
        """Load examples for training, validation and testing.

        See README.md for how to format data so it can be loaded here.
        NOTE: this converts everything to lower case.

        Args:
            data_dir (str): absolute path to dataset

        Returns:
            EditDataSplits
        """
        # load free words
        with codecs.open(join(data_dir, 'free.txt'), 'r', encoding='utf-8') as f:
            free = [line.strip().lower() for line in f]
            free_set = set(free)

        def examples_from_file(path):
            """Return list[EditExample] from file path."""
            examples = []

            # count total lines before loading
            total_lines = int(local('wc -l {}'.format(path), capture=True).split()[0])

            with codecs.open(path, 'r', encoding='utf-8') as f:
                for line in verboserate(f, desc='Reading data file.', total=total_lines):
                    src, trg = line.strip().lower().split('\t')
                    src_words = src.split(' ')
                    trg_words = trg.split(' ')
                    assert len(src_words) > 0
                    assert len(trg_words) > 0

                    if use_diff:
                        ex = EditExample.salient_diff(src_words, trg_words, free_set)
                    else:
                        ex = EditExample.whitelist_blacklist(src_words, trg_words)
                    examples.append(ex)
            return examples

        self.train = examples_from_file(join(data_dir, 'train.tsv'))
        self.valid = examples_from_file(join(data_dir, 'valid.tsv'))
        self.test = examples_from_file(join(data_dir, 'test.tsv'))
        self.free = free


class RandomState(object):
    def __init__(self):
        """Take a snapshot of random number generator state at this point in time.

        Only covers random, numpy.random and torch (CPU).
        """
        self.py = random.getstate()
        self.np = np.random.get_state()
        self.torch = torch.get_rng_state()

    def set_global(self):
        """Set all global random number generators to this state."""
        random.setstate(self.py)
        np.random.set_state(self.np)
        torch.set_rng_state(self.torch)


# TODO(kelvin): make this a classmethod on RandomState?
@contextmanager
def random_state(state):
    """Execute code inside this with-block by starting with the specified random state.

    Does not affect the state of random number generators outside this block.
    Not thread-safe.

    Args:
        state (RandomState)
    """
    old_state = RandomState()
    state.set_global()
    yield
    old_state.set_global()


# TODO(kelvin): reduce coupling with RandomState
@contextmanager
def random_seed(seed):
    """Execute code inside this with-block using the specified random seed.

    Sets the seed for random, numpy.random and torch (CPU).

    WARNING: torch GPU seeds are NOT set!

    Does not affect the state of random number generators outside this block.
    Not thread-safe.

    Args:
        seed (int)
    """
    state = RandomState()
    random.seed(seed)  # alter state
    np.random.seed(seed)
    torch.manual_seed(seed)
    yield
    state.set_global()


class TrainState(object):
    def __init__(self, editor, optimizer, train_steps, random_state, max_grad_norm):
        """Construct a snapshot of training state.

        Args:
            editor (Editor)
            optimizer (Optimizer)
            train_steps (int)
            random_state (RandomState)
            max_grad_norm (float): used for gradient clipping
        """
        self.editor = editor
        self.optimizer = optimizer
        self.train_steps = train_steps
        self.random_state = random_state
        self.max_grad_norm = max_grad_norm

    def update_random_state(self):
        """Store the latest random state."""
        self.random_state = RandomState()

    def increment_train_steps(self):
        self.train_steps += 1

    def track_grad_norms(self, grad_norm):
        # we will clip grad norm to be at most 2x the norm of anything we've tracked so far
        self.max_grad_norm = max(self.max_grad_norm, 2 * grad_norm)

    def save(self, checkpoints_dir):
        path = join(checkpoints_dir, '{}.checkpoint'.format(self.train_steps))
        gtd.io.makedirs(path)

        # save model
        torch.save(self.editor.state_dict(), join(path, 'editor'))
        torch.save(self.optimizer.state_dict(), join(path, 'optimizer'))

        # pickle remaining attributes
        d = {attr: getattr(self, attr) for attr in ['train_steps', 'random_state', 'max_grad_norm']}
        with open(join(path, 'metadata.p'), 'w') as f:
            pickle.dump(d, f)

    @classmethod
    def load(cls, path, editor, optimizer):
        with open(join(path, 'metadata.p'), 'r') as f:
            d = pickle.load(f)

        # load model
        optimizer.load_state_dict(torch.load(join(path, 'optimizer')))
        editor.load_state_dict(torch.load(join(path, 'editor')))
        train_state = TrainState(editor=editor, optimizer=optimizer, **d)
        return train_state


class EditTrainingRun(TorchTrainingRun):
    def __init__(self, config, save_dir):
        super(EditTrainingRun, self).__init__(config, save_dir)

        # extra dir for storing TrainStates where NaN was encountered
        self.workspace.add_dir('nan_checkpoints', 'nan_checkpoints')

        # reload train state (includes model)
        checkpoints_dir = self.workspace.checkpoints
        ckpt_num = self._get_latest_checkpoint_number(checkpoints_dir)
        if ckpt_num is None:
            print 'No checkpoint to reload. Initializing fresh.'
            self._train_state = self._initialize_train_state(config)
        else:
            print 'Reloaded checkpoint #{}'.format(ckpt_num)
            self._train_state = self._reload_train_state(checkpoints_dir, ckpt_num, config)

        # load data
        data_dir = join(data.workspace.root, config.dataset.path)
        self._examples = EditDataSplits(data_dir, config.dataset.use_diff)

    def train(self):
        self._train(self.config, self._train_state, self._examples, self.workspace, self.metadata, self.tb_logger)

    def reload(self, train_steps):
        """Reload the checkpoint that was saved after taking `train_steps` steps."""
        self._train_state = self._reload_train_state(self.workspace.checkpoints, train_steps, self.config)

    def evaluate(self, big_eval=True):
        """Run an evaluation without logging it."""
        self._evaluate(self.config, self.editor, self._examples, None, None, 0, big_eval, log=False)

    @property
    def editor(self):
        """Return the Editor."""
        return self._train_state.editor

    @property
    def examples(self):
        """Return data splits (EditDataSplits)."""
        return self._examples

    @classmethod
    def _build_editor(cls, config):
        """Build Editor.

        Args:
            config (Config): Editor config

        Returns:
            Editor
        """

        file_path = join(data.workspace.word_vectors, config.wvec_path)
        word_embeddings = SimpleEmbeddings.from_file(file_path, config.word_dim, vocab_size=config.vocab_size)
        word_embeddings = word_embeddings.with_special_tokens()
        source_token_embedder = TokenEmbedder(word_embeddings)
        target_token_embedder = TokenEmbedder(word_embeddings)

        if config.decoder_cell == 'SimpleDecoderCell':
            decoder_cell = SimpleDecoderCell(target_token_embedder, config.hidden_dim,
                                             config.word_dim, config.agenda_dim)
        elif config.decoder_cell == 'AttentionDecoderCell':
            decoder_cell = AttentionDecoderCell(target_token_embedder, config.agenda_dim,
                                                config.hidden_dim, config.hidden_dim,
                                                config.attention_dim, config.no_insert_delete_attn,
                                                num_layers=config.decoder_layers)
        else:
            raise ValueError('{} not implemented'.format(config.decoder_cell))

        editor = Editor(source_token_embedder, config.hidden_dim, config.agenda_dim, config.edit_dim, config.lamb_reg, config.norm_eps, config.norm_max, config.kill_edit, decoder_cell, config.encoder_layers)

        editor = try_gpu(editor)
        return editor

    @classmethod
    def _initialize_train_state(cls, config):
        """Set up all the state necessary to begin training."""
        with random_seed(config.seed):
            editor = cls._build_editor(config.editor)
            optimizer = optim.Adam(editor.parameters(), lr=config.optim.learning_rate)
            train_steps = 0
            max_grad_norm = 0
            random_state = RandomState()

        return TrainState(editor=editor, optimizer=optimizer, train_steps=train_steps,
                          random_state=random_state, max_grad_norm=max_grad_norm)

    @classmethod
    def _reload_train_state(cls, checkpoints_dir, train_steps, config):
        ckpt_path = join(checkpoints_dir, '{}.checkpoint'.format(train_steps))
        init_state = cls._initialize_train_state(config)
        editor, optimizer = init_state.editor, init_state.optimizer
        return TrainState.load(ckpt_path, editor, optimizer)

    @classmethod
    def _checkpoint_numbers(cls, checkpoints_dir):
        """Return the train steps at which checkpoints were saved (sorted ascending)."""
        dirs = [d for d in listdir(checkpoints_dir) if d.endswith('.checkpoint')]
        return sorted([int(d[:-11]) for d in dirs])

    @classmethod
    def _get_latest_checkpoint_number(cls, checkpoints_dir):
        """Return the train_steps of the latest saved checkpoint.

        If no checkpoints, return None.
        """
        nums = cls._checkpoint_numbers(checkpoints_dir)
        if len(nums) == 0:
            return None
        else:
            return max(nums)

    @classmethod
    def _finite_grads(cls, parameters):
        """Check that all parameter gradients are finite.

        Args:
            parameters (List[Parameter])

        Return:
            bool
        """
        for param in parameters:
            try:
                if not np.isfinite(param.grad.data.sum()):
                    return False
            except AttributeError:
                # allow some parameters not to have gradients and floating in the compute graph.
                pass
        return True

    @classmethod
    def _train(cls, config, train_state, examples, workspace, metadata, tb_logger):
        """Train a model.

        NOTE: modifies TrainState in place.
        - parameters of the Editor and Optimizer are updated
        - train_steps is updated
        - random number generator states are updated at every checkpoint

        Args:
            config (Config)
            train_state (TrainState): initial TrainState. Includes the Editor and Optimizer.
            examples (EditDataSplits)
            workspace (Workspace)
            metadata (Metadata)
            tb_logger (tensorboard_logger.Logger)
        """
        with random_state(train_state.random_state):
            editor = train_state.editor
            optimizer = train_state.optimizer
            noiser = EditNoiser(config.editor.ident_pr, config.editor.attend_pr)
            train_batches = similar_size_batches(examples.train, config.optim.batch_size)

            # test batching!
            editor.test_batch(noiser(train_batches[0]))

            while True:
                # TODO(kelvin): this shuffle and the position within the shuffle is not properly restored upon reload
                random.shuffle(train_batches)

                for batch in verboserate(train_batches, desc='Streaming training examples'):
                    # compute gradients
                    optimizer.zero_grad()
                    if config.editor.edit_dropout:
                        noised_batch = noiser(batch)
                    else:
                        noised_batch = batch
                    loss = editor.loss(noised_batch, draw_samples=config.editor.enable_vae)
                    loss.backward()

                    # clip gradients
                    if train_state.train_steps < 50:
                        # don't clip, just observe the gradient norm
                        grad_norm = clip_grad_norm(editor.parameters(), float('inf'), norm_type=2)
                        train_state.track_grad_norms(grad_norm)
                        metadata['max_grad_norm'] = train_state.max_grad_norm
                    else:
                        # clip according to the max allowed grad norm
                        grad_norm = clip_grad_norm(editor.parameters(), train_state.max_grad_norm)
                        # this returns the gradient norm BEFORE clipping

                    finite_grads = cls._finite_grads(editor.parameters())

                    # take a step if the grads are finite
                    if finite_grads:
                        optimizer.step()

                    # increment step count
                    train_state.increment_train_steps()

                    # somehow we encountered NaN
                    if not finite_grads:
                        # dump parameters
                        train_state.save(workspace.nan_checkpoints)

                        # dump offending example batch
                        examples_path = join(workspace.nan_checkpoints, '{}.examples'.format(train_state.train_steps))
                        with open(examples_path, 'w') as f:
                            pickle.dump(noised_batch, f)

                        print 'Gradient was NaN/inf on step {}.'.format(train_state.train_steps)

                        # if there were more than 5 NaNs in the last 10 steps, drop into the debugger
                        nan_steps = cls._checkpoint_numbers(workspace.nan_checkpoints)
                        recent_nans = [s for s in nan_steps if s > train_state.train_steps - 10]
                        if len(recent_nans) > 5:
                            print 'Too many NaNs encountered recently: {}. Entering debugger.'.format(recent_nans)
                            import pdb
                            pdb.set_trace()

                    # run periodic evaluation and saving
                    if train_state.train_steps % config.eval.eval_steps == 0:
                        cls._evaluate(config, editor, examples, metadata, tb_logger, train_state.train_steps, noiser, big_eval=False)
                        tb_logger.log_value('grad_norm', grad_norm, train_state.train_steps)

                    if train_state.train_steps % config.eval.big_eval_steps == 0:
                        cls._evaluate(config, editor, examples, metadata, tb_logger, train_state.train_steps, noiser, big_eval=True)

                    if train_state.train_steps % config.eval.save_steps == 0:
                        train_state.update_random_state()
                        train_state.save(workspace.checkpoints)

                    if train_state.train_steps >= config.optim.max_iters:
                        return

    @classmethod
    def _evaluate(cls, config, editor, examples, metadata, tb_logger, train_steps, noiser, big_eval, log=True):

        def log_value(name, value, step):
            if log:
                # log to both TensorBoard and metadata file
                tb_logger.log_value(name, value, step)
                metadata[name] = value

        def evaluate_on_examples(name, examples):
            # use more samples for big evaluation
            num_eval = config.eval.big_num_examples if big_eval else config.eval.num_examples
            big_str = 'big_' if big_eval else ''

            # compute metrics
            loss, avg_bleu, edit_traces = cls._compute_metrics(editor, examples, num_eval, noiser,
                                                               edit_dropout=config.editor.edit_dropout,
                                                               draw_samples=config.editor.enable_vae)

            # log
            log_value('loss_{}{}'.format(big_str, name), loss, train_steps)
            log_value('bleu_{}{}'.format(big_str, name), avg_bleu, train_steps)

            print '=== {}{} ==='.format(big_str, name)
            print 'loss: {}, bleu: {}'.format(loss, avg_bleu)

            # print traces for the small evaluation
            if not big_eval:
                for tr in edit_traces:
                    print tr

        print '===== STEP {} ====='.format(train_steps)
        evaluate_on_examples('train', examples.train)
        evaluate_on_examples('valid', examples.valid)

    @classmethod
    def _compute_metrics(cls, editor, examples, num_evaluate_examples, noiser,
                         batch_size=256, edit_dropout=False, draw_samples=False):
        with random_seed(0):
            sample = sample_if_large(examples, num_evaluate_examples, replace=False)
        if edit_dropout:
            noised_sample = noiser(sample)
        else:
            noised_sample = sample

        # compute loss and log to TensorBoard
        # need to break the sample into batches, in case the sample is too large to fit in GPU memory
        losses, weights = [], []
        for batch in chunks(noised_sample, batch_size):
            weights.append(len(batch))
            loss_var = editor.loss(batch, draw_samples)
            losses.append(loss_var.data[0])
        losses, weights = np.array(losses), np.array(weights)
        loss = np.sum(losses * weights) / np.sum(weights)  # weighted average

        # compute BLEU score and log to TensorBoard
        outputs, edit_traces = editor.edit(noised_sample)
        bleus = []
        for ex, output in izip(noised_sample, outputs):
            # outputs is a list(over batches)[ list(over beams) [ list(over tokens) [ unicode ] ] ] object.
            bleus.append(bleu(ex.target_words, output[0]))
        avg_bleu = np.mean(bleus)
        return loss, avg_bleu, edit_traces
