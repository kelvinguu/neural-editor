import numpy as np
import pytest

from gtd.ml.torch.recurrent import AdditionCell
from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.source_encoder import BidirectionalSourceEncoder
from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.torch.utils import assert_tensor_equal
from gtd.ml.vocab import SimpleVocab
from gtd.utils import Bunch


class TestBidirectionalSourceEncoder(object):
    @pytest.fixture
    def encoder(self):
        return BidirectionalSourceEncoder(1, 2, AdditionCell)

    @pytest.fixture
    def input_embeds_list(self):
        sequences = [
            [1, 2, 3],
            [8, 4, 2, 1, 1],
            [],
        ]

        # token 1 maps to embedding [1], 2 maps to [2] and so on...
        vocab = SimpleVocab([1, 2, 3, 4, 5, 6, 7, 8])
        array = np.expand_dims(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32), 1)
        token_embedder = TokenEmbedder(Bunch(vocab=vocab, array=array))

        seq_embeds = token_embedder.embed_seq_batch(SequenceBatch.from_sequences(sequences, vocab))
        return seq_embeds.split()

    def test_combined_states(self, encoder, input_embeds_list):
        states = encoder(input_embeds_list).combined_states

        # forward encoder is cumulatively summing from the left
        # backward encoder is cumulatively summing from the right

        # both encoders should ignore masked time steps
        assert_tensor_equal(states[0].values, [[1, 6],
                                               [8, 16],
                                               [0, 0],
                                               ])
        assert_tensor_equal(states[0].mask, [[1], [1], [0]])

        assert_tensor_equal(states[2].values, [[6, 3],
                                               [14, 4],
                                               [0, 0],
                                               ])
        assert_tensor_equal(states[2].mask, [[1], [1], [0]])

        assert_tensor_equal(states[3].values, [[6, 0],
                                               [15, 2],
                                               [0, 0],
                                               ])
        assert_tensor_equal(states[3].mask, [[0], [1], [0]])

    def test_final_states(self, encoder, input_embeds_list):
        forward, backward = encoder(input_embeds_list).final_states
        assert_tensor_equal(forward, [[6], [16], [0]])
        assert_tensor_equal(backward, [[6], [16], [0]])

