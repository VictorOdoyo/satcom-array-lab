import numpy as np


def db(value, floor_db=-200):
    from .core import finite

    x = finite(value, real=True)
    if np.any(x < 0) or not np.isfinite(floor_db):
        raise ValueError("nonnegative physical ratio and finite display floor required")
    return 10 * np.log10(np.maximum(x, 10 ** (floor_db / 10)))


def linear(decibels):
    from .core import finite

    value = 10 ** (finite(decibels, real=True) / 10)
    if not np.isfinite(value).all():
        raise ValueError("decibel conversion exceeds finite range")
    return value
