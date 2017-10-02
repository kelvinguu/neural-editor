import numpy as np
import pytest
import torch
from torch.nn import Embedding

from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.torch.utils import GPUVariable
from gtd.ml.torch.utils import assert_tensor_equal
from gtd.ml.vocab import SimpleVocab
from gtd.utils import Bunch


class TestTokenEmbedder(object):
    @pytest.fixture
    def embedder(self):
        vocab = SimpleVocab(['<unk>', '<start>', '<stop>'] + ['a', 'b', 'c'])
        arr = np.eye(len(vocab), dtype=np.float32)
        word_embeddings = Bunch(vocab=vocab, array=arr)
        return TokenEmbedder(word_embeddings)

    def test_embedding_from_array(self):
        emb = TokenEmbedder._embedding_from_array(np.array([[9, 9], [8, 7]], dtype=np.float32))
        assert isinstance(emb, Embedding)
        values = emb(GPUVariable(torch.LongTensor([[0, 0], [1, 0]])))

        assert_tensor_equal(values,
                            [
                                [[9, 9], [9, 9]],
                                [[8, 7], [9, 9]],
                            ])

    def test_embed_indices(self, embedder):
        indices = GPUVariable(torch.LongTensor([
            [0, 1],
            [2, 2],
            [4, 5],
        ]))

        embeds = embedder.embed_indices(indices)

        assert_tensor_equal(embeds, [
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
            [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
        ])

    def test_embed_tokens(self, embedder):
        tokens = ['b', 'c', 'c']
        embeds = embedder.embed_tokens(tokens)

        assert_tensor_equal(embeds, [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
        ])