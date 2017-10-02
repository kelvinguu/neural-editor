import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from gtd.ml.utils import temperature_smooth


def test_temperature_smooth():
    smooth = lambda probs, temp: temperature_smooth(np.array(probs, dtype=np.float32), temp)
    same = lambda x1, x2: assert_almost_equal(x1, x2, decimal=4)

    probs = [0., 0.2, 0.4, 0.4]
    third = 1./3
    correct = [0., third, third, third]
    same(smooth(probs, 100000), correct)

    # doesn't sum to 1
    with pytest.raises(ValueError):
        smooth([1, 2, 0], 1)

    # contains negative numbers
    with pytest.raises(ValueError):
        smooth([1, -1, 1], 1)

    # temperature = 0
    with pytest.raises(ValueError):
        probs = [0, 0.25, 0.75, 0]
        smooth(probs, 0)

    # temperature = inf
    with pytest.raises(ValueError):
        probs = [0, 0.25, 0.75, 0]
        smooth(probs, float('inf'))

    # temperature = 1
    probs = [0, 0.25, 0.75, 0]
    same(smooth(probs, 1), probs)  # shouldn't alter probs

    # contains 1
    probs = [1, 0, 0]
    same(smooth(probs, 10), probs)
    same(smooth(probs, 0.1), probs)

    a = np.exp(2)
    b = np.exp(3)

    probs = [0, a/(a+b), b/(a+b)]
    smoothed = smooth(probs, 11)

    a2 = np.exp(2. / 11)
    b2 = np.exp(3. / 11)
    correct = [0, a2/(a2+b2), b2/(a2+b2)]
    same(smoothed, correct)