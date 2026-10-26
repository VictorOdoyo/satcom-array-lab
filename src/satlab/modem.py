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


def acquire_tone(samples, sample_rate_hz, search_hz):
    from .core import positive

    s = finite(samples, ndim=1)
    fs = positive(sample_rate_hz)
    span = positive(search_hz)
    if len(s) < 16 or span >= fs / 2 or np.linalg.norm(s) == 0:
        raise ValueError("nonzero tone and sub-Nyquist search required")
    nfft = 4 * len(s)
    power = abs(np.fft.fftshift(np.fft.fft(s * np.hanning(len(s)), nfft))) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    eligible = np.flatnonzero(abs(freq) <= span)
    k = int(eligible[np.argmax(power[eligible])])
    delta = 0.0
    if 0 < k < nfft - 1:
        y = np.log(np.maximum(power[k - 1 : k + 2], np.finfo(float).tiny))
        den = y[0] - 2 * y[1] + y[2]
        if den != 0:
            delta = float(np.clip(0.5 * (y[0] - y[2]) / den, -0.5, 0.5))
    return {"offset_hz": float(freq[k] + delta * fs / nfft), "frequency_hz": freq, "power": power}
