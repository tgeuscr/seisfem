"""Source-time functions independent of finite elements."""

import numpy as np


def ricker(time, frequency: float, time_shift: float):
    """Dimensionless (1-2a²)exp(-a²), a=pi*f0*(t-t0), with t in s and f0 in Hz.

    f0 is the peak frequency of the amplitude spectrum of the untruncated wavelet.
    Peak value at t0 is +1. The function itself is not causally truncated.
    """
    if not np.isfinite(frequency) or frequency <= 0 or not np.isfinite(time_shift):
        raise ValueError("frequency must be finite and positive; time_shift must be finite")
    a = np.pi * frequency * (np.asarray(time) - time_shift)
    return (1 - 2 * a**2) * np.exp(-(a**2))
