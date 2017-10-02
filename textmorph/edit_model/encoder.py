from collections import namedtuple

import torch
from torch.nn import Module

from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.source_encoder import MultiLayerSourceEncoder
from gtd.ml.torch.utils import GPUVariable, NamedTupleLike
from textmorph.edit_model.agenda import AgendaMaker
from textmorph.edit_model.edit_encoder import EditEncoder

EncoderInput = namedtuple('EncoderInput',
                          ['source_words', 'insert_words', 'insert_exact_words', 'delete_words', 'delete_exact_words',
                           'edit_embed'])
"""
Args:
    source_words (SequenceBatch): of shape (batch_size, sequence_length)
    insert_words (SequenceBatch): of shape (batch_size, max_edits) this is the generative part, not always executable.
    insert_exact_words (SequenceBatch): of shape (batch_size, max_edits) these are concrete edits, should always be executable
    delete_words (SequenceBatch): of shape (batch_size, max_edits)
    delete_exact_words (SequenceBatch): of shape (batch_size, max_edits)
    edit_embed (Variable): of shape (batch_size, edit_dim).

Note:
    The `edit_embed` attribute is usually None. If it is not None, it will override any edit_embed produced
    by the EditEncoder.
"""

class EncoderOutput(namedtuple('EncoderOutput', ['source_embeds', 'insert_embeds', 'delete_embeds', 'agenda']), NamedTupleLike):
    pass
"""
Args:
    source_embeds (SequenceBatch): of shape (batch_size, seq_length, hidden_size)
    insert_embeds (SequenceBatch): of shape (batch_size, max_edits, word_dim) ONLY contains inserts for the exact part for attention.
    delete_embeds (SequenceBatch): of shape (batch_size, max_edits, word_dim)
    agenda (Variable): of shape (batch_size, agenda_dim)
"""


# TODO(kelvin):
# bidirectional LSTM
# make sure LSTM init properly
# also see TODOs in decoder.py

class Encoder(Module):
    def __init__(self, token_embedder, agenda_dim, edit_dim, hidden_dim, lamb_reg, norm_eps, norm_max, kill_edit, num_layers, rnn_cell_factory):
        """Construct Encoder.

        Args:
            token_embedder (TokenEmbedder)
            agenda_dim (int)
            edit_dim (int)
            hidden_dim (int)
            num_layers (int)
            rnn_cell_factory (Callable[[int, int], RNNCell): takes input_dim and output_dim as arguments.
        """
        super(Encoder, self).__init__()

        self.word_vocab = token_embedder.vocab
        self.token_embedder = token_embedder
        self.agenda_dim = agenda_dim
        self.edit_dim = edit_dim
        word_dim = token_embedder.embed_dim
        self.lamb_reg = lamb_reg
        self.kill_edit = kill_edit

        self.source_encoder = MultiLayerSourceEncoder(word_dim, hidden_dim, num_layers, rnn_cell_factory)
        self.edit_encoder = EditEncoder(word_dim, edit_dim, lamb_reg, norm_eps, norm_max)
        self.agenda_maker = AgendaMaker(self.source_encoder.hidden_dim, self.edit_dim, self.agenda_dim)

    def preprocess(self, source_words, insert_words, insert_exact_words, delete_words, delete_exact_words, edit_embed):
        """Preprocess.

        Args:
            source_words (list[list[unicode]]): a batch of source sequences
            insert_words (list[list[unicode]]): a batch of insert words
            insert_exact_words (list[list[unicode]]): a batch of insert words, used without noise
            delete_words (list[list[unicode]]): a batch of delete words
            delete_exact_words (list[list[unicode]]): a batch of delete words, used without noise
            edit_embed (np.ndarray | None): of shape (batch_size, edit_dim), or None.

        Returns:
            EncoderInput
        """
        return EncoderInput(
            SequenceBatch.from_sequences(source_words, self.word_vocab),
            SequenceBatch.from_sequences(insert_words, self.word_vocab, min_seq_length=1),
            SequenceBatch.from_sequences(insert_exact_words, self.word_vocab, min_seq_length=1),
            SequenceBatch.from_sequences(delete_words, self.word_vocab, min_seq_length=1),
            SequenceBatch.from_sequences(delete_exact_words, self.word_vocab, min_seq_length=1),
            edit_embed
        )
        # insert_words and delete_words will often be empty, but we still enforce a min_seq_length of 1 to avoid
        # creating Tensors with dimensions of 0, which some Torch function will choke on.

    def _vmfKL(self, k, d):
        return k*((sp.special.iv(d/2.0+1.0,k)\
                   + sp.special.iv(d/2.0,k)*d/(2.0*k))/sp.special.iv(d/2.0, k) - d/(2.0*k))\
                   + d * np.log(k)/2.0 - np.log(sp.special.iv(d/2.0,k)) \
                   - sp.special.loggamma(d/2+1) - d * np.log(2)/2

    def regularizer(self, encoder_input):
        """Compute and return per-batch regularizer.

        NOTE: not implemented.
        """
        return 0.

    def forward(self, encoder_input, draw_samples=False, draw_p = False):
        """Encode.

        Args:
            encoder_input (EncoderInput)
            draw_samples (bool) : flag for whether to add noise for variational approx. disable at test time.

        Returns:
            EncoderOutput
        """
        source_words = encoder_input.source_words
        source_word_embeds = self.token_embedder.embed_seq_batch(source_words)
        source_encoder_output = self.source_encoder(source_word_embeds.split())
        source_embeds_list = source_encoder_output.combined_states
        source_embeds = SequenceBatch.cat(source_embeds_list)
        # the final hidden states in both the forward and backward direction, concatenated
        source_embeds_final = torch.cat(source_encoder_output.final_states, 1)  # (batch_size, hidden_dim)

        insert_embeds = self.token_embedder.embed_seq_batch(encoder_input.insert_words)
        delete_embeds = self.token_embedder.embed_seq_batch(encoder_input.delete_words)

        insert_embeds_exact = self.token_embedder.embed_seq_batch(encoder_input.insert_exact_words)
        delete_embeds_exact = self.token_embedder.embed_seq_batch(encoder_input.delete_exact_words)

        insert_noisy_exact = self.edit_encoder.seq_batch_noise(insert_embeds_exact, draw_samples)
        delete_noisy_exact = self.edit_encoder.seq_batch_noise(delete_embeds_exact, draw_samples)

        batch_size, _ = source_embeds_final.size()

        if self.kill_edit:
                edit_embed = GPUVariable(torch.zeros(batch_size, self.edit_dim))
        else:
            if encoder_input.edit_embed is None:
                edit_embed = self.edit_encoder(insert_embeds, insert_embeds_exact,
                                                   delete_embeds, delete_embeds_exact,draw_samples, draw_p)
            else:
                # bypass the edit_encoder
                edit_embed = encoder_input.edit_embed

        agenda = self.agenda_maker(source_embeds_final, edit_embed)
        return EncoderOutput(source_embeds, insert_noisy_exact, delete_noisy_exact, agenda)

    def warp_edit_vec(self, edit_embed, encoder_input):
        """ Wrap a given edit vector and generate encoder outputs """
        source_words = encoder_input.source_words
        source_word_embeds = self.token_embedder.embed_seq_batch(source_words)
        insert_embeds = self.token_embedder.embed_seq_batch(encoder_input.insert_words)
        delete_embeds = self.token_embedder.embed_seq_batch(encoder_input.delete_words)

        insert_embeds_exact = self.token_embedder.embed_seq_batch(encoder_input.insert_exact_words)
        delete_embeds_exact = self.token_embedder.embed_seq_batch(encoder_input.delete_exact_words)

        source_encoder_output = self.source_encoder(source_word_embeds.split())
        source_embeds_list = source_encoder_output.combined_states
        source_embeds = SequenceBatch.cat(source_embeds_list)
        # the final hidden states in both the forward and backward direction, concatenated
        source_embeds_final = torch.cat(source_encoder_output.final_states, 1)  # (batch_size, hidden_dim)

        agenda = self.agenda_maker(source_embeds_final, edit_embed)
        return EncoderOutput(source_embeds, insert_embeds_exact, delete_embeds_exact, agenda)

    def generate_edits(self, encoder_input, norm):
        """ Draw uniform random vectors with given norm, and use as edit vector """
        source_words = encoder_input.source_words
        source_word_embeds = self.token_embedder.embed_seq_batch(source_words)
        insert_embeds = self.token_embedder.embed_seq_batch(encoder_input.insert_words)
        delete_embeds = self.token_embedder.embed_seq_batch(encoder_input.delete_words)

        insert_embeds_exact = self.token_embedder.embed_seq_batch(encoder_input.insert_exact_words)
        delete_embeds_exact = self.token_embedder.embed_seq_batch(encoder_input.delete_exact_words)

        source_encoder_output = self.source_encoder(source_word_embeds.split())
        source_embeds_list = source_encoder_output.combined_states
        source_embeds = SequenceBatch.cat(source_embeds_list)
        # the final hidden states in both the forward and backward direction, concatenated
        source_embeds_final = torch.cat(source_encoder_output.final_states, 1)  # (batch_size, hidden_dim)

        edit_encoded = self.edit_encoder(insert_embeds, delete_embeds)

        rand_vec = torch.randn(edit_encoded.shape())
        edit_embed = GPUVariable(rand_vec / torch.norm(rand_vec, 2, dim=1).expand_as(rand_vec) * norm)
        agenda = self.agenda_maker(source_embeds_final, edit_embed)
        return EncoderOutput(source_embeds, insert_embeds_exact, delete_embeds_exact, agenda)
