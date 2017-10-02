import numpy as np


def temperature_smooth(sampling_probs, temperature):
    """Smooth a discrete distribution by raising/lowering temperature.

    Args:
        sampling_probs (np.ndarray): 1D numpy array
        temperature (float)

    Returns:
        np.ndarray: 1D array of same shape as sampling_probs
    """
    if not isinstance(sampling_probs, np.ndarray):
        raise TypeError("sampling_probs must be numpy array.")

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    if not np.isfinite(temperature):
        raise ValueError("Temperature must be finite.")

    if abs(np.sum(sampling_probs) - 1.0) > 0.001:
        raise ValueError("sampling_probs must sum to 1.")

    if not np.all(sampling_probs >= 0):
        raise ValueError("sampling_probs must all be non-negative.")

    logits = np.log(sampling_probs)  # should be in range [-inf, 0]
    unnormalized = np.exp(logits / temperature)
    probs = unnormalized / np.sum(unnormalized)
    return probs