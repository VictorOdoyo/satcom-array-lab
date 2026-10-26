import numpy as np

from .core import finite, integer, positive


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


def sample(samples, positions):
    from scipy.interpolate import CubicSpline

    x = finite(samples, ndim=1)
    t = finite(positions, real=True)
    if len(x) < 4 or np.any(t < 0) or np.any(t > len(x) - 1):
        raise ValueError("interpolation positions must remain inside the waveform")
    return CubicSpline(np.arange(len(x)), x)(t)


def fractional_delay(samples, delay_samples=0.35, taps=33):
    x = finite(samples, ndim=1)
    count = integer(taps, 5)
    delay = float(finite(delay_samples, real=True))
    if count % 2 == 0 or abs(delay) > count // 2 - 2:
        raise ValueError("odd delay-filter length and interior fractional delay required")
    center = count // 2
    n = np.arange(count) - center
    h = np.sinc(n - delay) * np.kaiser(count, 8.6)
    h /= sum(h)
    return {"samples": np.convolve(x, h), "integer_delay": center, "fractional_delay": delay, "taps": h}


def clock_drift(samples, parts_per_million=0):
    x = finite(samples, ndim=1)
    ratio = 1 + float(finite(parts_per_million, real=True)) * 1e-6
    if not 0.99 < ratio < 1.01:
        raise ValueError("clock offset limited to one percent")
    positions = np.arange(int(np.floor((len(x) - 1) / ratio)) + 1) * ratio
    return {"samples": sample(x, positions), "source_positions": positions}


def acquire_timing(samples, pilots, samples_per_symbol, minimum_lag, maximum_lag, step_samples=0.125):
    x = finite(samples, ndim=1)
    p = finite(pilots, ndim=1)
    sps = integer(samples_per_symbol, 2)
    step = positive(step_samples)
    low, high = finite([minimum_lag, maximum_lag], real=True)
    if low < 0 or high < low or len(p) < 8 or high + (len(p) - 1) * sps > len(x) - 1:
        raise ValueError("bounded pilot timing search required")
    lags = np.arange(low, high + step * 0.25, step)
    score = []
    for lag in lags:
        observed = sample(x, lag + np.arange(len(p)) * sps)
        power = np.vdot(observed, observed).real
        score.append(float(abs(np.vdot(p, observed)) ** 2 / (np.vdot(p, p).real * power)) if power > 0 else 0)
    best = int(np.argmax(score))
    return {"lag_samples": float(lags[best]), "lags": lags, "score": np.array(score)}
