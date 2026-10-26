import numpy as np

from .core import finite


def qpsk(bits):
    b = finite(bits, real=True, ndim=1)
    if len(b) % 2 or np.any((b != 0) & (b != 1)):
        raise ValueError("even binary bit stream required")
    pair = b.reshape(-1, 2)
    return ((1 - 2 * pair[:, 0]) + 1j * (1 - 2 * pair[:, 1])) / np.sqrt(2)


def detect(symbols):
    s = finite(symbols, ndim=1)
    return np.column_stack([s.real < 0, s.imag < 0]).astype(np.uint8).ravel()


def awgn(symbols, esn0_db, rng):
    from .link import linear

    s = finite(symbols, ndim=1)
    sigma = np.sqrt(1 / (2 * linear(esn0_db)))
    return s + sigma * (rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape))


def carrier(samples, sample_rate_hz, offset_hz, phase_rad=0, start_s=0):
    from .core import positive

    s = finite(samples, ndim=1)
    t = start_s + np.arange(len(s)) / positive(sample_rate_hz)
    return s * np.exp(
        1j * (2 * np.pi * float(finite(offset_hz, real=True)) * t + float(finite(phase_rad, real=True)))
    )
