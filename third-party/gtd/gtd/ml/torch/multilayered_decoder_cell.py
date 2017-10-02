from collections import namedtuple

import torch
from torch.nn import LSTMCell, Linear, Parameter, Softmax

from gtd.ml.torch.decoder_cell import DecoderCell, DecoderCellOutput, RNNState, RNNInput
from gtd.ml.torch.recurrent import tile_state, gated_update
from gtd.ml.torch.utils import GPUVariable, try_gpu


class MultilayeredRNNState(namedtuple('MultilayeredRNNState', ['hs', 'cs']), RNNState):
    pass


class MultilayeredRNNInput(namedtuple('MultilayeredRNNInput', ['x', 'agenda']), RNNInput):
    pass


class MultilayeredDecoderCell(DecoderCell):
    def __init__(self, token_embedder, hidden_dim, input_dim, agenda_dim, num_layers):
        super(MultilayeredDecoderCell, self).__init__()
        self.linear = Linear(hidden_dim, input_dim)
        self.h0 = Parameter(torch.zeros(hidden_dim))
        self.c0 = Parameter(torch.zeros(hidden_dim))
        self.softmax = Softmax()
        self.token_embedder = token_embedder
        self.num_layers = num_layers

        self.rnn_cells = []
        for layer in range(num_layers):
            in_dim = (input_dim + agenda_dim) if layer == 0 else hidden_dim # inputs to first layer are word vectors
            out_dim = hidden_dim
            rnn_cell = LSTMCell(in_dim, out_dim)
            self.add_module('decoder_layer_{}'.format(layer), rnn_cell)
            self.rnn_cells.append(rnn_cell)

    @property
    def rnn_state_type(self):
        return MultilayeredRNNState

    @property
    def rnn_input_type(self):
        return MultilayeredRNNInput

    def initialize(self, batch_size):
        h = tile_state(self.h0, batch_size)
        c = tile_state(self.c0, batch_size)
        return MultilayeredRNNState([h] * self.num_layers, [c] * self.num_layers)

    def forward(self, rnn_state, rnn_input, advance):
        x = torch.cat([rnn_input.x, rnn_input.agenda], 1)
        hs, cs = [], []
        for layer in range(self.num_layers):
            rnn_cell = self.rnn_cells[layer]

            # collect the h, c belonging to the previous time-step at the corresponding depth
            h_prev_t, c_prev_t = rnn_state.hs[layer], rnn_state.cs[layer]

            # forward pass and masking
            h, c = rnn_cell(x, (h_prev_t, c_prev_t))
            h = gated_update(h_prev_t, h, advance)
            c = gated_update(c_prev_t, c, advance)
            hs.append(h)
            cs.append(c)

            if layer == 0:
                x = h  # no skip connection on the first layer
            else:
                x = x + h

        query = self.linear(x)
        word_vocab = self.token_embedder.vocab
        word_embeds = self.token_embedder.embeds
        vocab_logits = torch.mm(query, word_embeds.t())  # (batch_size, vocab_size)
        vocab_probs = self.softmax(vocab_logits)

        rnn_state = MultilayeredRNNState(hs, cs)

        return DecoderCellOutput(rnn_state, vocab=word_vocab, vocab_probs=vocab_probs)