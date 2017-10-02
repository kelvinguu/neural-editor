import numpy as np
import pytest
import torch
from gtd.ml.torch.utils import GPUVariable
from gtd.ml.torch.utils import assert_tensor_equal

from gtd.ml.torch.seq_batch import SequenceBatch, SequenceBatchElement
from gtd.ml.vocab import SimpleVocab


class TestSequenceBatch(object):
    @pytest.fixture
    def sequences(self):
        return [
            ['a', 'b', 'b', 'c'],
            ['c'],
            [],
        ]

    @pytest.fixture
    def vocab(self):
        return SimpleVocab(['<unk>', 'a', 'b', 'c', '<start>', '<stop>'])

    def test_from_sequences(self, sequences, vocab):
        seq_batch = SequenceBatch.from_sequences(sequences, vocab)

        assert_tensor_equal(seq_batch.values,
                            np.array([
                                [1, 2, 2, 3],
                                [3, 0, 0, 0],
                                [0, 0, 0, 0],
                            ], dtype=np.int32))

        assert_tensor_equal(seq_batch.mask,
                            np.array([
                                [1, 1, 1, 1],
                                [1, 0, 0, 0],
                                [0, 0, 0, 0],
                            ], dtype=np.float32))

    def test_min_seq_length(self, vocab):
        seq_batch = SequenceBatch.from_sequences([[], [], []], vocab, min_seq_length=2)
        assert_tensor_equal(seq_batch.values, np.zeros((3, 2)))
        assert_tensor_equal(seq_batch.mask, np.zeros((3, 2)))

    def test_mask_validation(self):
        mask = GPUVariable(torch.FloatTensor([[1, 0, 0, 0],
                                              [1, 1, 0, 0],
                                              [1, 1, 1, 0]]))

        values = mask  # just set values = mask, since it doesn't matter

        # should not raise any errors
        SequenceBatch(values, mask)

        non_binary_mask = GPUVariable(torch.FloatTensor([[1, 0, 0, 0],
                                                         [1, 1.2, 0, 0],
                                                         [1, 1, 1, 0]]))

        with pytest.raises(ValueError):
            SequenceBatch(mask, non_binary_mask)

        non_left_justified_mask = GPUVariable(torch.FloatTensor([[1, 0, 0, 1],
                                                                 [1, 1, 0, 0],
                                                                 [1, 1, 1, 0]]))

        with pytest.raises(ValueError):
            SequenceBatch(mask, non_left_justified_mask)

    def test_split(self):
        input_embeds = GPUVariable(torch.LongTensor([
            # batch item 1
            [
                [1, 2], [2, 3], [5, 6]
            ],
            # batch item 2
            [
                [4, 8], [3, 5], [0, 0]
            ],
        ]))

        input_mask = GPUVariable(torch.FloatTensor([
            [1, 1, 1],
            [1, 1, 0],
        ]))

        sb = SequenceBatch(input_embeds, input_mask)

        elements = sb.split()
        input_list = [e.values for e in elements]
        mask_list = [e.mask for e in elements]

        assert len(input_list) == 3
        assert_tensor_equal(input_list[0], [[1, 2], [4, 8]])
        assert_tensor_equal(input_list[1], [[2, 3], [3, 5]])
        assert_tensor_equal(input_list[2], [[5, 6], [0, 0]])

        assert len(mask_list) == 3
        assert_tensor_equal(mask_list[0], [[1], [1]])
        assert_tensor_equal(mask_list[1], [[1], [1]])
        assert_tensor_equal(mask_list[2], [[1], [0]])

    def test_cat(self):
        x1 = SequenceBatchElement(
            GPUVariable(torch.FloatTensor([
                [[1, 2], [3, 4]],
                [[8, 2], [9, 0]]])),
            GPUVariable(torch.FloatTensor([
                [1],
                [1]
            ])))
        x2 = SequenceBatchElement(
            GPUVariable(torch.FloatTensor([
                [[-1, 20], [3, 40]],
                [[-8, 2], [9, 10]]])),
            GPUVariable(torch.FloatTensor([
                [1],
                [0]
            ])))
        x3 = SequenceBatchElement(
            GPUVariable(torch.FloatTensor([
                [[-1, 20], [3, 40]],
                [[-8, 2], [9, 10]]])),
            GPUVariable(torch.FloatTensor([
                [0],
                [0]
            ])))

        result = SequenceBatch.cat([x1, x2, x3])

        assert_tensor_equal(result.values,
                            [
                                [[[1, 2], [3, 4]], [[-1, 20], [3, 40]], [[-1, 20], [3, 40]]],
                                [[[8, 2], [9, 0]], [[-8, 2], [9, 10]], [[-8, 2], [9, 10]]],
                            ])

        assert_tensor_equal(result.mask,
                            [
                                [1, 1, 0],
                                [1, 0, 0]
                            ])

    @pytest.fixture
    def some_seq_batch(self):
        values = GPUVariable(torch.FloatTensor([
            [[1, 2], [4, 5], [4, 4]],
            [[0, 4], [43, 5], [-1, 20]],
            [[-1, 20], [43, 5], [0, 0]],
        ]))
        mask = GPUVariable(torch.FloatTensor([
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]))
        return SequenceBatch(values, mask)

    def test_weighted_sum(self, some_seq_batch):
        weights = GPUVariable(torch.FloatTensor([
            [0.5, 0.3, 0],
            [0.8, 0.2, 0],
            [0, 0, 0],
        ]))
        result = SequenceBatch.weighted_sum(some_seq_batch, weights)

        # [1, 2] * 0.5 + [4, 5] * 0.3 = [0.5 + 1.2, 1 + 1.5] = [1.7, 2.5]
        # [0, 4] * 0.8 = [0, 3.2]
        # 0

        # Weights on entries where mask[i, j] = 0 get ignored, as desired.
        assert_tensor_equal(result, [
            [1.7, 2.5],
            [0, 3.2],
            [0, 0],
        ])

    def test_reduce_sum(self, some_seq_batch):
        result = SequenceBatch.reduce_sum(some_seq_batch)

        assert_tensor_equal(result, [
            [5, 7],
            [0, 4],
            [0, 0],
        ])

    def test_reduce_mean(self, some_seq_batch):
        result = SequenceBatch.reduce_mean(some_seq_batch, allow_empty=True)

        assert_tensor_equal(result, [
            [2.5, 3.5],
            [0, 4],
            [0, 0]
        ])

        with pytest.raises(ValueError):
            SequenceBatch.reduce_mean(some_seq_batch, allow_empty=False)

    def test_reduce_prod(self, some_seq_batch):
        result = SequenceBatch.reduce_prod(some_seq_batch)
        assert_tensor_equal(result, [
            [4, 10],
            [0, 4],
            [1, 1]
        ])

    def test_reduce_max(self, some_seq_batch):

        with pytest.raises(ValueError):
            # should complain about empty sequence
            SequenceBatch.reduce_max(some_seq_batch)

        values = GPUVariable(torch.FloatTensor([
            [[1, 2], [4, 5], [4, 4]],  # actual max is in later elements, but shd be suppressed by mask
            [[0, -4], [43, -5], [-1, -20]],  # note that all elements in 2nd dim are negative
        ]))
        mask = GPUVariable(torch.FloatTensor([
            [1, 0, 0],
            [1, 1, 0],
        ]))
        seq_batch = SequenceBatch(values, mask)
        result = SequenceBatch.reduce_max(seq_batch)

        assert_tensor_equal(result, [
            [1, 2],
            [43, -4],
        ])

    def test_embed(self):
        sequences = [
            [],
            [1, 2, 3],
            [3, 3],
            [2]
        ]

        vocab = SimpleVocab([0, 1, 2, 3, 4])
        indices = SequenceBatch.from_sequences(sequences, vocab)

        embeds = GPUVariable(torch.FloatTensor([
            [0, 0],
            [2, 2],   # 1
            [3, 4],   # 2
            [-10, 1], # 3
            [11, -1]  # 4
        ]))

        embedded = SequenceBatch.embed(indices, embeds)

        correct = np.array([
            [[0, 0], [0, 0], [0, 0]],
            [[2, 2], [3, 4], [-10, 1]],
            [[-10, 1], [-10, 1], [0, 0]],
            [[3, 4], [0, 0], [0, 0]]
        ], dtype=np.float32)
        assert_tensor_equal(embedded.values, correct)