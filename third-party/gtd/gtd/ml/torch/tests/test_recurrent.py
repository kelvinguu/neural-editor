import torch
from gtd.ml.torch.utils import GPUVariable

from gtd.ml.torch.recurrent import tile_state, gated_update
from gtd.ml.torch.utils import assert_tensor_equal


def test_tile_state():
    h = GPUVariable(torch.FloatTensor([1, 2, 3]))
    h_tiled = tile_state(h, 3)
    assert_tensor_equal(h_tiled, [[1, 2, 3], [1, 2, 3], [1, 2, 3]])


def test_gated_update():
    h = GPUVariable(torch.FloatTensor([
        [1, 2, 3],
        [4, 5, 6],
    ]))
    h_new = GPUVariable(torch.FloatTensor([
        [-1, 2, 3],
        [4, 8, 0],
    ]))
    update = GPUVariable(torch.FloatTensor([[0], [1]]))  # only update the second row

    out = gated_update(h, h_new, update)

    assert_tensor_equal(out, [
        [1, 2, 3],
        [4, 8, 0]
    ])