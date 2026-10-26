import numpy as np

from .core import finite, integer


def rrc(samples_per_symbol=8, span_symbols=10, rolloff=0.25):
    sps = integer(samples_per_symbol, 2)
    span = integer(span_symbols, 2)
    beta = float(finite(rolloff, real=True))
    if not 0 <= beta <= 1 or span * sps % 2:
        raise ValueError("bounded rolloff and even total pulse span required")
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    if beta == 0:
        h = np.sinc(t)
    else:
        h = np.zeros(len(t))
        for k, value in enumerate(t):
            if abs(value) < 1e-12:
                h[k] = 1 + beta * (4 / np.pi - 1)
            elif abs(abs(value) - 1 / (4 * beta)) < 1e-12:
                h[k] = (
                    beta
                    / np.sqrt(2)
                    * (
                        (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                    )
                )
            else:
                h[k] = (
                    np.sin(np.pi * value * (1 - beta)) + 4 * beta * value * np.cos(np.pi * value * (1 + beta))
                ) / (np.pi * value * (1 - (4 * beta * value) ** 2))
    return h / np.linalg.norm(h)


def shape(symbols, pulse, samples_per_symbol=8):
    symbols = finite(symbols, ndim=1)
    h = finite(pulse, ndim=1)
    sps = integer(samples_per_symbol, 2)
    if len(symbols) == 0 or len(h) == 0:
        raise ValueError("nonempty symbols and pulse required")
    upsampled = np.zeros((len(symbols) - 1) * sps + 1, complex)
    upsampled[::sps] = symbols
    return np.convolve(upsampled, h)


def matched(samples, pulse):
    x = finite(samples, ndim=1)
    h = finite(pulse, ndim=1)
    if len(x) == 0 or len(h) == 0:
        raise ValueError("nonempty waveform and matched pulse required")
    return np.convolve(x, h.conj()[::-1])
