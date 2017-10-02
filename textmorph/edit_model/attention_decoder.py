from itertools import izip

import numpy as np
import torch
from torch.nn import LSTMCell, Linear, Parameter, Softmax

from collections import namedtuple
from gtd.ml.torch.attention import Attention, AttentionOutput, DummyAttention
from gtd.ml.torch.decoder_cell import DecoderCell, DecoderCellOutput, RNNState, RNNInput
from gtd.ml.torch.recurrent import gated_update, tile_state
from gtd.ml.torch.utils import GPUVariable
from gtd.utils import UnicodeMixin
from gtd.ml.torch.decoder import RNNContextCombiner


class AttentionContextCombiner(RNNContextCombiner):
    def __call__(self, encoder_output, x):
        return AttentionRNNInput(x=x, agenda=encoder_output.agenda, source_embeds=encoder_output.source_embeds, insert_embeds=encoder_output.insert_embeds, delete_embeds=encoder_output.delete_embeds)

class AttentionDecoderCell(DecoderCell):
    def __init__(self, token_embedder, agenda_dim, decoder_dim, encoder_dim, attn_dim, no_insert_delete_attn, num_layers):
        super(AttentionDecoderCell, self).__init__()

        input_dim = token_embedder.embed_dim
        self.num_layers = num_layers

        # see definition of `x_augment` in `forward` method
        # we augment the input to each RNN layer with 3 attention contexts + the agenda
        augment_dim = encoder_dim + input_dim + input_dim + agenda_dim

        self.rnn_cells = []
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else decoder_dim  # first layer takes word vectors
            out_dim = decoder_dim
            rnn_cell = LSTMCell(in_dim + augment_dim, out_dim)
            self.add_module('decoder_layer_{}'.format(layer), rnn_cell)
            self.rnn_cells.append(rnn_cell)

        # see definition of `z` in `forward` method
        # to predict words, we condition on the hidden state h + 3 attention contexts
        z_dim = decoder_dim + encoder_dim + 2 * input_dim
        if no_insert_delete_attn:
            z_dim = decoder_dim + encoder_dim

        self.vocab_projection_pos = Linear(z_dim, input_dim)  # TODO(kelvin): these big params may need regularization
        self.vocab_projection_neg = Linear(z_dim, input_dim)
        self.relu = torch.nn.ReLU()

        self.h0 = Parameter(torch.zeros(decoder_dim))
        self.c0 = Parameter(torch.zeros(decoder_dim))
        self.vocab_softmax = Softmax()

        self.source_attention = Attention(encoder_dim, decoder_dim, attn_dim)
        if not no_insert_delete_attn:
            self.insert_attention = Attention(input_dim, decoder_dim, attn_dim)
            self.delete_attention = Attention(input_dim, decoder_dim, attn_dim)
        else:
            self.insert_attention = DummyAttention(input_dim, decoder_dim, attn_dim)
            self.delete_attention = DummyAttention(input_dim, decoder_dim, attn_dim)

        self.token_embedder = token_embedder
        self.no_insert_delete_attn = no_insert_delete_attn

    def initialize(self, batch_size):
        h = tile_state(self.h0, batch_size)
        c = tile_state(self.c0, batch_size)

        # no initial weights, context is just zero vector
        init_attn = lambda attention: AttentionOutput(None, GPUVariable(torch.zeros(batch_size, attention.memory_dim)))

        return AttentionRNNState([h] * self.num_layers, [c] * self.num_layers, init_attn(self.source_attention),
                        init_attn(self.insert_attention), init_attn(self.delete_attention))

    def forward(self, rnn_state, decoder_cell_input, advance):
        dci = decoder_cell_input
        mask = advance

        # this will be concatenated to x at every layer
        # we are conditioning on the attention from the previous time step and the agenda from the encoder
        x_augment = torch.cat([rnn_state.source_attn.context,
                               rnn_state.insert_attn.context,
                               rnn_state.delete_attn.context,
                               dci.agenda], 1)

        hs, cs = [], []
        x = dci.x  # input word vector
        for layer in range(self.num_layers):
            rnn_cell = self.rnn_cells[layer]
            old_h, old_c = rnn_state.hs[layer], rnn_state.cs[layer]
            rnn_input = torch.cat([x, x_augment], 1)
            h, c = rnn_cell(rnn_input, (old_h, old_c))
            h = gated_update(old_h, h, mask)
            c = gated_update(old_c, c, mask)
            hs.append(h)
            cs.append(c)

            if layer == 0:
                x = h  # no skip connection on the first layer
            else:
                x = x + h

        # compute attention using bottom layer
        source_attn = self.source_attention(dci.source_embeds, hs[0])
        insert_attn = self.insert_attention(dci.insert_embeds, hs[0])
        delete_attn = self.delete_attention(dci.delete_embeds, hs[0])
        if not self.no_insert_delete_attn:
            z = torch.cat([x, source_attn.context, insert_attn.context, delete_attn.context], 1)
        else:
            z = torch.cat([x, source_attn.context], 1)

        # has shape (batch_size, decoder_dim + encoder_dim + input_dim + input_dim)

        vocab_query_pos = self.vocab_projection_pos(z)
        vocab_query_neg = self.vocab_projection_neg(z)
        word_vocab = self.token_embedder.vocab
        word_embeds = self.token_embedder.embeds
        vocab_logit_pos = self.relu(torch.mm(vocab_query_pos, word_embeds.t())) # (batch_size, vocab_size)
        vocab_logit_neg = self.relu(torch.mm(vocab_query_neg, word_embeds.t()))  # (batch_size, vocab_size)
        vocab_probs = self.vocab_softmax(vocab_logit_pos - vocab_logit_neg)
        # TODO(kelvin): prevent model from putting probability on UNK

        rnn_state = AttentionRNNState(hs, cs, source_attn, insert_attn, delete_attn)

        return DecoderCellOutput(rnn_state, vocab=word_vocab, vocab_probs=vocab_probs)

    def rnn_state_type(self):
        return AttentionRNNState

    def rnn_input_type(self):
        return AttentionRNNInput

class AttentionRNNState(namedtuple('AttentionRNNState', ['hs','cs','source_attn','insert_attn','delete_attn']), RNNState):
    """
    Attributes:
    hs (list[Variable]): a list of the hidden states for each layer of a multi-layer RNN.
        Each Variable has shape (batch_size, hidden_dim).
    cs (list[Variable]): a list of the cell states for each layer of a multi-layer RNN
        Each Variable has shape (batch_size, hidden_dim).
    source_attn (AttentionOutput)
    insert_attn (AttentionOutput)
    delete_attn (AttentionOutput)
    """
    pass

class AttentionRNNInput(namedtuple('AttentionRNNInput', ['x','agenda','source_embeds','insert_embeds','delete_embeds']), RNNInput):
    """
Attributes:
    x (Variable): of shape (batch_size, word_dim), embedding of word generated at previous time step
    agenda (Variable): of shape (batch_size, agenda_dim)
    source_embeds (SequenceBatch): of shape (batch_size, source_seq_length, hidden_size)
    insert_embeds (SequenceBatch): of shape (batch_size, max_edits, embed_dim)
    delete_embeds (SequenceBatch): of shape (batch_size, max_edits, embed_dim)
    """
    pass

        

class AttentionTrace(UnicodeMixin):
    __slots__ = ['name', 'tokens', 'attention_weights']

    def __init__(self, name, tokens, attention_weights):
        """Construct AttentionTrace.

        Args:
            name (unicode): name of attention mechanism
            tokens (list[unicode])
            attention_weights (np.ndarray): a 1D array. May be longer than len(tokens) due to batching.
        """
        assert len(attention_weights.shape) == 1

        # any attention weights exceeding length of tokens should be zero
        for i in range(len(tokens), len(attention_weights)):
            assert attention_weights[i] == 0

        self.name = name
        self.tokens = tokens
        self.attention_weights = attention_weights

    def __unicode__(self):
        total_mass = np.sum(self.attention_weights)
        s = u' '.join(u'{}[{:.2f}]'.format(t, w) for t, w in izip(self.tokens, self.attention_weights))
        return u'{:10}[{:.2f}]: {}'.format(self.name, total_mass, s)
