import pytest
from gtd.ml.torch.utils import assert_tensor_equal

from gtd.ml.torch.alignments import Alignments


class TestAlignments(object):
    @pytest.fixture
    def source_words(self):
        return [
            ['a', 'c', 'b', 'c'],
            ['1', '3', '2', '2', '2'],
            [],
        ]

    @pytest.fixture
    def target_words(self):
        return [
            ['c', 'z', 'b', 'c'],
            ['1', 'c'],
            ['2', '4'],
        ]

    @pytest.fixture
    def aligns(self, source_words, target_words):
        return Alignments(source_words, target_words)

    def test(self, aligns):
        assert_tensor_equal(aligns.indices,
                            [
                                [[1, 3], [0, 0], [2, 0], [1, 3]],
                                [[0, 0], [0, 0], [0, 0], [0, 0]],
                                [[0, 0], [0, 0], [0, 0], [0, 0]],
                            ])

        assert_tensor_equal(aligns.mask,
                            [
                                [[1, 1], [0, 0], [1, 0], [1, 1]],
                                [[1, 0], [0, 0], [0, 0], [0, 0]],
                                [[0, 0], [0, 0], [0, 0], [0, 0]],
                            ])

    def test_split(self, aligns):
        items = aligns.split()
        assert len(items) == 4

        assert_tensor_equal(items[0].values,
                            [
                                [1, 3],
                                [0, 0],
                                [0, 0]
                            ])

        assert_tensor_equal(items[0].mask,
                            [
                                [1, 1],
                                [1, 0],
                                [0, 0]
                            ])

        assert_tensor_equal(items[2].values,
                            [
                                [2, 0],
                                [0, 0],
                                [0, 0]
                            ])

        assert_tensor_equal(items[2].mask,
                            [
                                [1, 0],
                                [0, 0],
                                [0, 0]
                            ])