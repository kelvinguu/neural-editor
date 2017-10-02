from collections import namedtuple

import torch
from torch.nn import LSTMCell, Linear, Parameter, Softmax

from gtd.ml.torch.decoder_cell import DecoderCell, DecoderCellOutput, RNNState, RNNInput
from gtd.ml.torch.recurrent import tile_state, gated_update


class SimpleRNNState(namedtuple('SimpleRNNState', ['h', 'c']), RNNState):
    pass


class SimpleRNNInput(namedtuple('SimpleRNNInput', ['x', 'agenda']), RNNInput):
    pass


class SimpleDecoderCell(DecoderCell):
    def __init__(self, token_embedder, hidden_dim, input_dim, agenda_dim):
        super(SimpleDecoderCell, self).__init__()
        self.rnn_cell = LSTMCell(input_dim + agenda_dim, hidden_dim)
        self.linear = Linear(hidden_dim, input_dim)
        self.h0 = Parameter(torch.zeros(hidden_dim))
        self.c0 = Parameter(torch.zeros(hidden_dim))
        self.softmax = Softmax()
        self.token_embedder = token_embedder

    @property
    def rnn_state_type(self):
        return SimpleRNNState

    @property
    def rnn_input_type(self):
        return SimpleRNNInput

    def initialize(self, batch_size):
        h = tile_state(self.h0, batch_size)
        c = tile_state(self.c0, batch_size)
        return SimpleRNNState(h, c)

    def forward(self, rnn_state, rnn_input, advance):
        rnn_input_embed = torch.cat([rnn_input.x, rnn_input.agenda], 1)
        h, c = self.rnn_cell(rnn_input_embed, (rnn_state.h, rnn_state.c))

        # don't update if sequence has terminated
        h = gated_update(rnn_state.h, h, advance)
        c = gated_update(rnn_state.c, c, advance)

        query = self.linear(h)
        word_vocab = self.token_embedder.vocab
        word_embeds = self.token_embedder.embeds
        vocab_logits = torch.mm(query, word_embeds.t())  # (batch_size, vocab_size)
        vocab_probs = self.softmax(vocab_logits)

        # no attention over source, insert and delete embeds
        rnn_state = SimpleRNNState(h, c)

        return DecoderCellOutput(rnn_state, vocab=word_vocab, vocab_probs=vocab_probs)