from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple

import torch
from torch.nn import Module

from gtd.ml.torch.utils import NamedTupleLike


# marker class
class RNNState(NamedTupleLike):
    __slots__ = []


# marker class
class RNNInput(NamedTupleLike):
    __slots__ = []


PredictionBatch = namedtuple('PredictionBatch', ['probs', 'vocab'])
"""
Attributes:
    probs (np.ndarray): of shape (batch_size, vocab_size)
    vocab (Vocab)
"""


class DecoderCellOutput(namedtuple('DecoderCellOutput', ['rnn_state', 'vocab', 'vocab_probs'])):
    """
    Attributes:
        rnn_state (RNNState)
        vocab (Vocab)
        vocab_probs (Variable): of shape (batch_size, vocab_size)
    """

    def loss(self, target_word):
        """Compute loss for this time step.

        Args:
            target_word (Variable): LongTensor of shape (batch_size,)

        Returns:
            loss (Variable): of shape (batch_size,)
        """
        target_prob = torch.gather(self.vocab_probs, 1, target_word.unsqueeze(1)).squeeze(1)  # (batch_size,)
        assert len(target_prob.size()) == 1

        loss = -torch.log(target_prob + 1e-45)  # negative log-likelihood
        # added 1e-45 to prevent loss from being -infinity in the case where probs is close to 0
        return loss

    @property
    def predictions(self):
        """Return a PredictionBatch.

        Returns:
            PredictionBatch
        """
        return PredictionBatch(self.vocab_probs.data.cpu().numpy(), self.vocab)


class DecoderCell(Module):
    __metaclass__ = ABCMeta

    @abstractproperty
    def rnn_state_type(self):
        pass

    @abstractproperty
    def rnn_input_type(self):
        pass

    @abstractmethod
    def initialize(self, batch_size):
        """Return initial RNNState.

        Args:
            batch_size (int)

        Returns:
            RNNState
        """
        raise NotImplementedError

    def forward(self, rnn_state, rnn_input, advance):
        """Advance the decoder by one step.

        Args:
            rnn_state (RNNState): the previous RNN state.
            rnn_input (RNNInput): any inputs at this time step.
            advance (Variable): of shape (batch_size, 1). The RNN should advance on example i iff mask[i] == 1.

        Returns:
            DecoderCellOutput
        """
        raise NotImplementedError