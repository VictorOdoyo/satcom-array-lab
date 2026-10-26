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


def test_pilot_gain_phase():
    p = m.qpsk([0, 0, 0, 1, 1, 0, 1, 1])
    gain = 0.4 * np.exp(0.7j)
    fit = m.pilot_fit(gain * p, p)
    assert fit["gain"] == pytest.approx(gain)
    assert np.allclose(fit["equalized"], p)


def test_quality_analytic():
    p = m.qpsk([0, 0, 1, 1])
    q = m.quality(1.1 * p, p, [0, 0, 1, 1])
    assert q["bit_errors"] == 0
    assert q["evm_rms"] == pytest.approx(0.1)


def test_wilson_zero_errors():
    lo, hi = m.wilson(0, 100)
    assert lo == pytest.approx(0, abs=1e-15)
    assert 0.03 < hi < 0.04
    with pytest.raises(ValueError):
        m.wilson(2, 1)


def test_burst_recovery():
    result = m.burst(18, 1200, 3, 1024)
    assert result["ber"] < 0.005
    assert abs(result["residual_offset_hz"]) < 1
    assert result["evm_rms"] < 0.2


def test_qpsk_analytic_ber_consistency():
    from satlab.link import qpsk_ber

    rng = np.random.default_rng(73)
    bits = rng.integers(0, 2, 300000)
    received = m.awgn(m.qpsk(bits), 6, rng)
    count = int(np.count_nonzero(m.detect(received) != bits))
    lo, hi = m.wilson(count, len(bits), z=4)
    assert lo < float(qpsk_ber(6)) < hi
