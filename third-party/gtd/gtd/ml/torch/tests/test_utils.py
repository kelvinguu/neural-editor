import pytest
import torch

from gtd.ml.torch.utils import expand_dims_for_broadcast, assert_tensor_equal, is_binary


def test_expand_dims_for_broadcast():
    low_tensor = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    high_tensor = torch.zeros(2, 3, 8, 1)

    new_tensor = expand_dims_for_broadcast(low_tensor, high_tensor)

    assert new_tensor.size() == (2, 3, 1, 1)

    assert_tensor_equal(new_tensor.squeeze(), low_tensor)

    with pytest.raises(AssertionError):
        bad_tensor = torch.zeros(2, 4, 8, 1)  # prefix doesn't match
        expand_dims_for_broadcast(low_tensor, bad_tensor)


def test_is_binary():
    t1 = torch.FloatTensor([0, 1, 0, 0])
    t2 = torch.FloatTensor([0, -1, 0, 0])
    t3 = torch.FloatTensor([0, 0.1, 0.2, 0])
    assert is_binary(t1)
    assert not is_binary(t2)
    assert not is_binary(t3)