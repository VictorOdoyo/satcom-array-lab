import numpy as np
import pytest

from satlab import modem as m


def test_qpsk_mapping():
    symbols = m.qpsk([0, 0, 0, 1, 1, 1, 1, 0])
    assert np.allclose(abs(symbols), 1)
    assert np.allclose(symbols * np.sqrt(2), [1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j])
    with pytest.raises(ValueError):
        m.qpsk([0, 2])


def test_detect_roundtrip():
    bits = np.random.default_rng(7).integers(0, 2, 200)
    assert np.array_equal(m.detect(m.qpsk(bits)), bits)


def test_awgn_variance():
    noise = m.awgn(np.zeros(100000, dtype=complex), 10, np.random.default_rng(4))
    assert np.mean(abs(noise) ** 2) == pytest.approx(0.1, rel=0.015)
    assert abs(np.mean(noise)) < 0.004


def test_carrier_clock():
    s = m.carrier(np.ones(4), 8, 2, np.pi / 2)
    assert np.allclose(s, [1j, -1, -1j, 1])
    assert np.allclose(m.carrier(np.ones(2), 8, 2, start_s=0.5), s[:2] / 1j)


def test_tone_acquisition():
    s = m.carrier(np.ones(1024), 20000, 1732.3, 0.7)
    noisy = m.awgn(s, 0, np.random.default_rng(2))
    assert abs(m.acquire_tone(noisy, 20000, 4000)["offset_hz"] - 1732.3) < 3
