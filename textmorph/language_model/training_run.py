import codecs
import random
from os import makedirs
from os.path import dirname, realpath, join, exists

import numpy as np
from torch import optim
from torch.autograd import Variable

from gtd.chrono import verboserate
from gtd.io import num_lines
from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.torch.training_run import TorchTrainingRun
from gtd.ml.torch.utils import similar_size_batches
from gtd.ml.torch.utils import try_gpu
from gtd.ml.training_run import TrainingRuns
from gtd.ml.vocab import SimpleEmbeddings
from gtd.utils import sample_if_large, chunks
from textmorph import data
from textmorph.language_model.language_model import LanguageModel, \
    NoisyLanguageModel


class LMTrainingRun(TorchTrainingRun):

    def __init__(self, config, save_dir):
        super(LMTrainingRun, self).__init__(config, save_dir)

        # reload model
        model, optimizer = self._build_model(config)
        self._train_state = self.checkpoints.load_latest(model, optimizer)

        # load data
        data_dir = join(data.workspace.root, config.dataset.path)
        self._examples = DataSplits(data_dir, model.vocab)

        # store interpolations
        self._interps_dir = join(dirname(self.checkpoints._path), 'interps')
        if not exists(self._interps_dir):
            makedirs(self._interps_dir)

    def train(self):
        self._train(self.config, self._train_state, self._examples)

    def evaluate(self):
        self._evaluate(self.config, self._train_state)

    def _build_model(cls, config):
        file_path = join(data.workspace.word_vectors, config.model.wvec_path)
        word_embeddings = SimpleEmbeddings.from_file(
            file_path, config.model.word_dim, vocab_size=config.model.vocab_size)
        word_embeddings = word_embeddings.with_special_tokens()
        token_embedder = TokenEmbedder(word_embeddings)

        model = None
        if config.model.type == 0: # regular language model
            model = LanguageModel(token_embedder, config.model.hidden_dim,
                                  config.model.agenda_dim, config.model.num_layers, cls._make_logger())
        elif config.model.type == 1: # SVAE
            model = NoisyLanguageModel(token_embedder, config.model.hidden_dim,
                                       config.model.agenda_dim, config.model.num_layers,
                                       config.model.kl_weight_steps, config.model.kl_weight_rate,
                                       config.model.kl_weight_cap, config.model.dci_keep_rate,
                                       cls._make_logger())
        assert model is not None

        model = try_gpu(model)
        optimizer = optim.Adam(
            model.parameters(), lr=config.optim.learning_rate)
        return model, optimizer

    def _train(cls, config, train_state, examples):
        model = train_state.model
        optimizer = train_state.optimizer
        train_batches = similar_size_batches(
            examples.train, config.optim.batch_size, size=lambda ex: len(ex))

        while True:
            random.shuffle(train_batches)
            i = 0  # cannot enumerate(verboserate(...))
            for batch in verboserate(train_batches, desc='Streaming training examples'):
                loss = model.loss(batch, cls._train_state.train_steps)
                cls._take_grad_step(train_state, loss)
                if (i % 100) == 0:
                    cls.evaluate()
                if (i % 1000) == 0:
                    if config.model.type == 1: # SVAE
                        # write interpolations to file
                        fname = "interps_batches_{}".format(i)
                        num_ex = 10
                        a_idx = np.random.randint(len(batch), size=num_ex)
                        b_idx = np.random.randint(len(batch), size=num_ex)
                        interps = []
                        for a, b in zip(a_idx, b_idx):
                            ex_a = batch[a]
                            ex_b = batch[b]
                            interpolation = model._interpolate_examples(ex_a, ex_b)
                            interpolation_repr = []
                            interpolation_repr.append(" ".join(ex_a))
                            interpolation_repr.extend(
                                [" ".join(ex) for ex in interpolation])
                            interpolation_repr.append(" ".join(ex_b))
                            interps.append(interpolation_repr)
                        with open(join(cls._interps_dir, fname), 'w') as fout:
                            data = "\n\n".join(["\n".join(ex) for ex in interps])
                            fout.write(data.encode('utf-8'))
                if (i % 5000) == 0:
                    cls.checkpoints.save(train_state)
                i += 1

    def _make_logger(cls, should_log=(lambda ts: ts % 100 == 0)):
        """
        Logger is a closure that can be used by model for logging its internal state
        """
        logger = lambda name, x, ts, tb=cls.tb_logger, sl=should_log: tb.log_value(
            name, x, ts) if sl(ts) else None
        return logger

    def _evaluate(cls, config, train_state):

        # evaluate was called, logging must occur
        logger = cls._make_logger(lambda ts: True)

        def report(name, examples):
            loss, pp = cls._compute_metrics(
                train_state.model, train_state.train_steps, examples, config.eval.big_num_examples, config.optim.batch_size)

            ts = train_state.train_steps
            logger('loss_e_{}'.format(name), loss, ts)
            logger('loss_10_{}'.format(name), loss, ts)
            logger('pp_{}'.format(name), pp, ts)

        report('train', cls._examples.train)
        report('valid', cls._examples.valid)

    def _compute_metrics(cls, model, ts, examples, eval_size=1000, batch_size=256):

        examples_ = sample_if_large(examples, max_size=eval_size)

        losses, weights = [], []
        for batch in chunks(examples_, batch_size):
            # compute loss
            batch_loss = model.loss(batch, ts)
            losses.append(batch_loss.data[0])
            weights.append(len(batch))
        losses, weights = np.array(losses), np.array(weights)
        loss = np.sum(losses * weights) / np.sum(weights)

        # compute perplexity
        entropy = 0.0
        num_words = 0
        for batch in chunks(examples_, batch_size):
            # change base
            losses = model.per_instance_losses(batch)  # -log_e_x
            losses = losses.data.cpu().numpy()
            losses_log_2 = losses / np.log(2.0)

            # normalize log_p by sentence length
            lengths = np.array([len(ex) + 1 for ex in batch])
            entropy += np.sum(losses_log_2)
            num_words += sum(lengths)

        pp = 2.0 ** (1.0 / num_words * entropy)

        return round(loss, 5), round(pp, 55)


class LMTrainingRuns(TrainingRuns):

    def __init__(self, check_commit=True):
        data_dir = data.workspace.lm_runs
        # root of the Git repo
        src_dir = dirname(dirname(dirname(realpath(__file__))))
        run_factory = LMTrainingRun
        super(LMTrainingRuns, self).__init__(
            data_dir, src_dir, run_factory, check_commit=check_commit)


class DataSplits(object):
    """
    Attributes:
        train (list[list[unicode]])
        valid (list[list[unicode]])
        test (list[list[unicode]])
    """

    def __init__(self, data_dir, vocab):
        """Load examples for training, validation and testing.

        See README.md for how to format data so it can be loaded here.
        NOTE: this converts everything to lower case.

        Args:
            data_dir (str): absolute path to dataset

        Returns:
            DataSplits
        """
        def examples_from_file(path):
            """Return list[list[unicode]] from file path."""
            examples = []

            # count total lines before loading
            total_lines = num_lines(path)

            with codecs.open(path, 'r', encoding='utf-8') as f:
                for line in verboserate(f, desc='Reading data file.', total=total_lines):
                    ex = line.strip().lower()
                    ex_words = ex.split(' ')
                    assert len(ex_words) > 0
                    examples.append(ex_words)

            return examples

        self.train = examples_from_file(join(data_dir, 'train.txt'))
        self.valid = examples_from_file(join(data_dir, 'valid.txt'))
        self.test = examples_from_file(join(data_dir, 'test.txt'))
