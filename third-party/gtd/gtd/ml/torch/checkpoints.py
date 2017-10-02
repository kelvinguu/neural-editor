import cPickle as pickle
from os import listdir
import os
from os.path import join

import torch

import gtd
from gtd.ml.torch.utils import RandomState


class TrainState(object):
    def __init__(self, model, optimizer, train_steps, random_state, max_grad_norm):
        """Construct a snapshot of training state.

        Args:
            model (Module)
            optimizer (Optimizer)
            train_steps (int)
            random_state (RandomState)
            max_grad_norm (float): used for gradient clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.train_steps = train_steps
        self.random_state = random_state
        self.max_grad_norm = max_grad_norm

    def increment_train_steps(self):
        self.train_steps += 1

    def track_grad_norms(self, grad_norm):
        # we will clip grad norm to be at most 2x the norm of anything we've tracked so far
        self.max_grad_norm = max(self.max_grad_norm, 2 * grad_norm)

    def save(self, path):
        gtd.io.makedirs(path)

        # Store the latest random state
        self.random_state = RandomState()

        # save model
        torch.save(self.model.state_dict(), join(path, 'model'))
        torch.save(self.optimizer.state_dict(), join(path, 'optimizer'))

        # pickle remaining attributes
        d = {attr: getattr(self, attr) for attr in ['train_steps', 'random_state', 'max_grad_norm']}
        with open(join(path, 'metadata.p'), 'w') as f:
            pickle.dump(d, f)

    @classmethod
    def load(cls, path, model, optimizer):
        with open(join(path, 'metadata.p'), 'r') as f:
            d = pickle.load(f)

        # load model
        optimizer.load_state_dict(torch.load(join(path, 'optimizer')))
        model.load_state_dict(torch.load(join(path, 'model')))
        train_state = TrainState(model=model, optimizer=optimizer, **d)
        return train_state

    @classmethod
    def initialize(cls, model, optimizer):
        train_steps = 0
        max_grad_norm = 0
        random_state = RandomState()
        return TrainState(model=model, optimizer=optimizer, train_steps=train_steps,
                          random_state=random_state, max_grad_norm=max_grad_norm)


class Checkpoints(object):
    def __init__(self, checkpoints_dir):
        self._path = checkpoints_dir

    @property
    def checkpoint_numbers(self):
        """Return the train steps at which checkpoints were saved (sorted ascending)."""
        dirs = [d for d in listdir(self._path) if d.endswith('.checkpoint')]
        return sorted([int(d[:-11]) for d in dirs])  # '.checkpoint' is 11 characters

    @property
    def latest_checkpoint_number(self):
        """Return the train_steps of the latest saved checkpoint.

        If no checkpoints, return None.
        """
        nums = self.checkpoint_numbers
        if len(nums) == 0:
            return None
        else:
            return max(nums)

    def load(self, train_steps, model, optimizer):
        """Load the checkpoint for a particular training step.

        Args:
            model (Module)
            optimizer (Optimizer)

        Returns:
            TrainState
        """
        ckpt_path = join(self._path, '{}.checkpoint'.format(train_steps))
        if not os.path.exists(ckpt_path):
            raise ValueError('Checkpoint #{} does not exist.'.format(train_steps))
        return TrainState.load(ckpt_path, model, optimizer)

    def save(self, train_state):
        """Save TrainState."""
        ckpt_path = join(self._path, '{}.checkpoint'.format(train_state.train_steps))
        train_state.save(ckpt_path)

    def load_latest(self, model, optimizer):
        """Load the latest checkpoint.
        
        If there are no checkpoints, return a freshly initialized Checkpoint.
        
        Args:
            model (Module)
            optimizer (Optimizer)

        Returns:
            TrainState
        """
        ckpt_num = self.latest_checkpoint_number
        if ckpt_num is None:
            print 'No checkpoint to reload. Initializing fresh.'
            return TrainState.initialize(model, optimizer)
        else:
            train_state = self.load(self.latest_checkpoint_number, model, optimizer)
            print 'Reloaded checkpoint #{}'.format(ckpt_num)
            return train_state