from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple
from itertools import izip

import torch
from gtd.ml.torch.recurrent import tile_state, gated_update
from torch.nn import Module
from torch.nn import Parameter

from gtd.ml.torch.seq_batch import SequenceBatchElement


class SourceEncoder(Module):
    __metaclass__ = ABCMeta

    @abstractproperty
    def hidden_dim(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_embeds_list):
        """Embed a source sequence.

        Args:
            input_embeds_list (list[SequenceBatchElement]): where each element is of shape (batch_size, input_dim)

        Returns:
            hidden_states_list (list[SequenceBatchElement]) where each element is (batch_size, hidden_dim)
        """
        raise NotImplementedError


class SimpleSourceEncoder(SourceEncoder):
    def __init__(self, rnn_cell):
        """

        Args:
            rnn_cell (DecoderCell)
        """
        super(SimpleSourceEncoder, self).__init__()
        self.rnn_cell = rnn_cell
        hidden_dim = self.rnn_cell.hidden_size
        self.h0 = Parameter(torch.zeros(hidden_dim))
        self.c0 = Parameter(torch.zeros(hidden_dim))
        self._hidden_dim = hidden_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def forward(self, input_embeds_list):
        """

        Args:
            input_embeds_list (list[SequenceBatchElement]): where each element is of shape (batch_size, input_dim)

        Returns:
            hidden_states_list (list[SequenceBatchElement]) where each element is (batch_size, hidden_dim)
        """
        batch_size = input_embeds_list[0].values.size()[0]

        h = tile_state(self.h0, batch_size)  # (batch_size, hidden_dim)
        c = tile_state(self.c0, batch_size)  # (batch_size, hidden_dim)

        hidden_states_list = []

        for t, x in enumerate(input_embeds_list):
            # x.values has shape (batch_size, input_dim)
            # x.mask has shape (batch_size, 1)
            h_new, c_new = self.rnn_cell(x.values, (h, c))
            h = gated_update(h, h_new, x.mask)
            c = gated_update(c, c_new, x.mask)
            hidden_states_list.append(SequenceBatchElement(h, x.mask))

        return hidden_states_list


class BidirectionalSourceEncoder(SourceEncoder):
    def __init__(self, input_dim, hidden_dim, rnn_cell_factory):
        super(BidirectionalSourceEncoder, self).__init__()

        if hidden_dim % 2 != 0:
            raise ValueError('hidden_dim must be even for BidirectionalSourceEncoder.')
        self._hidden_dim = hidden_dim

        build_encoder = lambda: SimpleSourceEncoder(rnn_cell_factory(input_dim, hidden_dim / 2))
        self.forward_encoder = build_encoder()
        self.backward_encoder = build_encoder()

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def forward(self, input_embeds_list):
        """Compute bidirectional RNN embeddings.
        
        Args:
            input_embeds_list (list[SequenceBatchElement])

        Returns:
            forward_states (list[SequenceBatchElement]): ordered left to right
            backward_states (list[SequenceBatchElement]): ordered left to right
        """
        reverse = lambda seq: list(reversed(seq))
        forward_states = self.forward_encoder(input_embeds_list)
        backward_states = reverse(self.backward_encoder(reverse(input_embeds_list)))
        return BidirectionalEncoderOutput(forward_states, backward_states)


class BidirectionalEncoderOutput(namedtuple('BidirectionalEncoderOutput', ['forward_states', 'backward_states'])):
    """
    Attributes:
        forward_states (list[SequenceBatchElement]): ordered left to right
        backward_states (list[SequenceBatchElement]): ordered left to right
    """
    @property
    def combined_states(self):
        """Concatenates forward and backward hidden states: [forward; backward].
        
        Returns:
            combined_states (list[SequenceBatchElement]): ordered left to right
        """
        combined_states = [SequenceBatchElement(torch.cat([f.values, b.values], 1), f.mask)
                           for f, b in izip(self.forward_states, self.backward_states)]
        return combined_states

    @property
    def final_states(self):
        """Return the final forward and backward states.

        Returns:
            forward_state (Variable): right-most forward state, of shape (batch_size, hidden_dim)
            backward_state (Variable): left-most backward state, of shape (batch_size, hidden_dim)
        """
        return self.forward_states[-1].values, self.backward_states[0].values


# TODO(kelvin): test this
class MultiLayerSourceEncoder(SourceEncoder):
    def __init__(self, input_dim, hidden_dim, num_layers, rnn_cell_factory):
        """

        Args:
            input_dim (int)
            hidden_dim (int)
            num_layers (int)
            rnn_cell_factory (Callable[[int, int], RNNCell): takes input_dim and output_dim as arguments.
        """
        super(MultiLayerSourceEncoder, self).__init__()
        self.layers = []
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            out_dim = hidden_dim
            encoder = BidirectionalSourceEncoder(in_dim, out_dim, rnn_cell_factory)
            self.add_module('encoder_layer_{}'.format(layer), encoder)
            self.layers.append(encoder)

    @property
    def hidden_dim(self):
        return self.layers[-1].hidden_dim

    def forward(self, input_embeds_list):
        """

        Args:
            input_embeds_list (list[SequenceBatchElement]): where each element is of shape (batch_size, input_dim)

        Returns:
            hidden_states_list (list[SequenceBatchElement]) where each element is (batch_size, hidden_dim)
        """
        for i, layer in enumerate(self.layers):
            if i == 0:
                prev_hidden_states = input_embeds_list
            else:
                prev_hidden_states = [SequenceBatchElement(torch.cat([f.values, b.values], 1), f.mask)
                                      for f, b in izip(forward_states, backward_states)]

            new_forward_states, new_backward_states = layer(prev_hidden_states)

            if i == 0:
                # no skip connections here, because dimensions don't match
                forward_states, backward_states = new_forward_states, new_backward_states
            else:
                # add residuals to previous hidden states
                add_residuals = lambda a_list, b_list: [SequenceBatchElement(a.values + b.values, a.mask)
                                                        for a, b in izip(a_list, b_list)]

                forward_states = add_residuals(forward_states, new_forward_states)
                backward_states = add_residuals(backward_states, new_backward_states)

        return BidirectionalEncoderOutput(forward_states, backward_states)