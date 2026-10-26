import numpy as np

from .core import finite


def qpsk(bits):
    b = finite(bits, real=True, ndim=1)
    if len(b) % 2 or np.any((b != 0) & (b != 1)):
        raise ValueError("even binary bit stream required")
    pair = b.reshape(-1, 2)
    return ((1 - 2 * pair[:, 0]) + 1j * (1 - 2 * pair[:, 1])) / np.sqrt(2)
