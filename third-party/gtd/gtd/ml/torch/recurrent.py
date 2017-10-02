import torch
from gtd.ml.torch.utils import GPUVariable
from torch.nn import Module

from gtd.ml.torch.utils import conditional


def tile_state(h, batch_size):
    """Tile a given hidden state batch_size times.

    Args:
        h (Variable): a single hidden state of shape (hidden_dim,)
        batch_size (int)

    Returns:
        a Variable of shape (batch_size, hidden_dim)
    """
    tiler = GPUVariable(torch.ones(batch_size, 1))
    return torch.mm(tiler, h.unsqueeze(0))  # (batch_size, hidden_size)


def gated_update(h, h_new, update):
    """If update == 1.0, return h_new; if update == 0.0, return h.

    Applies this logic to each element in a batch.

    Args:
        h (Variable): of shape (batch_size, hidden_dim)
        h_new (Variable): of shape (batch_size, hidden_dim)
        update (Variable): of shape (batch_size, 1).

    Returns:
        Variable: of shape (batch_size, hidden_dim)

    """
    batch_size, hidden_dim = h.size()
    gate = update.expand(batch_size, hidden_dim)
    return conditional(gate, h_new, h)


class AdditionCell(Module):
    """Just add the input vector to the hidden state vector."""

    def __init__(self, input_dim, hidden_dim):
        super(AdditionCell, self).__init__()
        self.W = GPUVariable(torch.eye(input_dim, hidden_dim))
        # truncates input if input_dim > hidden_dim
        # pads with zeros if input_dim < hidden_dim
        self.hidden_size = hidden_dim

    def forward(self, x, hc):
        h, c = hc
        h = x.mm(self.W) + h
        return h, c