import random
from contextlib import contextmanager

import numpy as np
from numpy.testing import assert_array_almost_equal
from torch import _TensorBase, torch
from torch.autograd import Variable

from gtd.utils import chunks


def conditional(b, x, y):
    """Conditional operator for PyTorch.

    Args:
        b (FloatTensor): with values that are equal to 0 or 1
        x (FloatTensor): of same shape as b
        y (FloatTensor): of same shape as b

    Returns:
        z (FloatTensor): of same shape as b. z[i] = x[i] if b[i] == 1 else y[i]
    """
    return b * x + (1 - b) * y


def to_numpy(x):
    if isinstance(x, Variable):
        x = x.data  # unwrap Variable

    if isinstance(x, _TensorBase):
        x = x.cpu().numpy()
    return x


def assert_tensor_equal(x, y, decimal=6):
    assert_array_almost_equal(to_numpy(x), to_numpy(y), decimal=decimal)


def expand_dims_for_broadcast(low_tensor, high_tensor):
    """Expand the dimensions of a lower-rank tensor, so that its rank matches that of a higher-rank tensor.

    This makes it possible to perform broadcast operations between low_tensor and high_tensor.

    Args:
        low_tensor (Tensor): lower-rank Tensor with shape [s_0, ..., s_p]
        high_tensor (Tensor): higher-rank Tensor with shape [s_0, ..., s_p, ..., s_n]

    Note that the shape of low_tensor must be a prefix of the shape of high_tensor.

    Returns:
        Tensor: the lower-rank tensor, but with shape expanded to be [s_0, ..., s_p, 1, 1, ..., 1]
    """
    low_size, high_size = low_tensor.size(), high_tensor.size()
    low_rank, high_rank = len(low_size), len(high_size)

    # verify that low_tensor shape is prefix of high_tensor shape
    assert low_size == high_size[:low_rank]

    new_tensor = low_tensor
    for _ in range(high_rank - low_rank):
        new_tensor = torch.unsqueeze(new_tensor, len(new_tensor.size()))

    return new_tensor


def is_binary(t):
    """Check if values of t are binary.
    
    Args:
        t (Tensor|Variable)

    Returns:
        bool
    """
    if isinstance(t, Variable):
        t = t.data  # convert Variable to Tensor

    binary = (t == 0) | (t == 1)  # ByteTensor, should be all 1's
    all_binary = torch.prod(binary)  # int, should be 1
    return all_binary == 1


def similar_size_batches(examples, batch_size, size=lambda x: len(x.target_words)):
    """Create similar-sized batches of EditExamples.

    By default, elements with similar len('source_words') are batched together.
    See editor.py / EditExample.

    Args:
        examples (list[EditExample])
        batch_size (int)
        size (Callable[[EditExample], int])

    Returns:
        list[list[EditExample]]
    """
    assert batch_size >= 1
    sorted_examples = sorted(examples, key=size)
    batches = list(chunks(sorted_examples, batch_size))
    random.shuffle(batches)  # in-place

    # report savings
    suboptimal_batches = list(chunks(examples, batch_size))

    total_cost = lambda batches: batch_size * sum(max(size(b) for b in batch) for batch in batches)
    naive_cost = total_cost(suboptimal_batches)
    improved_cost = total_cost(batches)
    optimal_cost = sum(size(ex) for ex in examples)

    print 'Optimized batches: reduced cost from {naive} (naive) to {improved} ({reduction}% reduction).\n' \
          'Optimal (batch_size=1) would be {optimal}.'.format(naive=naive_cost, improved=improved_cost,
                                                              reduction=float(naive_cost - improved_cost) / naive_cost,
                                                              optimal=optimal_cost)

    return batches


def print_module_parameters(m, depth=0):
    """Print out all parameters of a module."""
    tabs = '\t' * depth
    for p_name, p in m._parameters.items():
        print tabs + p_name
    for c_name, c in m.named_children():
        print tabs + c_name
        print_module_parameters(c, depth + 1)


_GPUS_EXIST = True  # True by default

def try_gpu(x):
    """Try to put a Variable/Tensor/Module on GPU."""
    global _GPUS_EXIST

    if _GPUS_EXIST:
        try:
            return x.cuda()
        except (AssertionError, RuntimeError):
            # actually, GPUs don't exist
            print 'No GPUs detected. Sticking with CPUs.'
            _GPUS_EXIST = False
            return x
    else:
        return x


def GPUVariable(data):
    return try_gpu(Variable(data, requires_grad=False))


class RandomState(object):
    def __init__(self):
        """Take a snapshot of random number generator state at this point in time.

        Only covers random, numpy.random and torch (CPU).
        """
        self.py = random.getstate()
        self.np = np.random.get_state()
        self.torch = torch.get_rng_state()

    def set_global(self):
        """Set all global random number generators to this state."""
        random.setstate(self.py)
        np.random.set_state(self.np)
        torch.set_rng_state(self.torch)


@contextmanager
def random_state(state):
    """Execute code inside this with-block by starting with the specified random state.

    Does not affect the state of random number generators outside this block.
    Not thread-safe.

    Args:
        state (RandomState)
    """
    old_state = RandomState()
    state.set_global()
    yield
    old_state.set_global()


@contextmanager
def random_seed(seed):
    """Execute code inside this with-block using the specified random seed.

    Sets the seed for random, numpy.random and torch (CPU).

    WARNING: torch GPU seeds are NOT set!

    Does not affect the state of random number generators outside this block.
    Not thread-safe.

    Args:
        seed (int)
    """
    state = RandomState()
    random.seed(seed)  # alter state
    np.random.seed(seed)
    torch.manual_seed(seed)
    yield
    state.set_global()


class NamedTupleLike(object):
    __slots__ = []
