import numpy as np
import pytest

from satlab import waveform as m


def test_rrc_symmetry_energy_and_singularities():
    for beta in [0, 0.25, 0.5, 1]:
        h = m.rrc(8, 10, beta)
        assert np.all(np.isfinite(h)) and np.allclose(h, h[::-1])
        assert h @ h == pytest.approx(1)


def test_shaped_single_symbol_energy():
    h = m.rrc()
    wave = m.shape([1j], h)
    assert np.vdot(wave, wave).real == pytest.approx(1)
    assert np.allclose(wave, 1j * h)


def test_matched_peak_and_isi():
    h = m.rrc(8, 12, 0.25)
    r = m.matched(m.shape([1], h), h)
    center = len(h) - 1
    assert r[center] == pytest.approx(1)
    assert max(abs(r[center + np.arange(1, 5) * 8])) < 0.01


def test_fractional_samples_of_polynomial():
    t = np.arange(20.0)
    x = t**3 + 1j * t**2
    positions = np.array([2.25, 8.75, 15.5])
    assert np.allclose(m.sample(x, positions), positions**3 + 1j * positions**2)
    with pytest.raises(ValueError):
        m.sample(x, [-1])


def test_zero_fractional_delay_is_integer_shift():
    x = np.array([1.0, 2, 3, 4])
    r = m.fractional_delay(x, 0)
    assert np.allclose(r["samples"][16:20], x, atol=1e-12)
    assert sum(r["taps"]) == pytest.approx(1)


def test_clock_grid():
    x = np.arange(100, dtype=float) + 0j
    r = m.clock_drift(x, 1000)
    assert len(r["samples"]) < len(x)
    assert np.allclose(r["samples"].real, r["source_positions"])


def test_pilot_timing_acquisition():
    from satlab.modem import qpsk

    bits = np.random.default_rng(3).integers(0, 2, 256)
    p = qpsk(bits)
    h = m.rrc()
    wave = m.matched(m.shape(p, h), h)
    r = m.acquire_timing(wave, p, 8, 78, 82)
    assert abs(r["lag_samples"] - 80) <= 0.125


def test_iq_real_matrix():
    x = np.array([1.0, 1j])
    y = m.iq_imbalance(x, 0.5, 0.2, 0.1 + 0.2j)
    assert y[0] == pytest.approx(1 + 0.1 + 1j * (0.5 * np.sin(0.2) + 0.2))
    assert y[1] == pytest.approx(0.1 + 1j * (0.5 * np.cos(0.2) + 0.2))


def test_iq_pilot_fit():
    from satlab.modem import qpsk

    p = qpsk(np.tile([0, 0, 0, 1, 1, 0, 1, 1], 10))
    r = m.iq_imbalance(p, 0.8, 0.1, 0.04 + 0.02j)
    fit = m.fit_iq(r, p)
    assert max(abs(fit["residual"])) < 1e-12
    assert fit["coefficients"][2] == pytest.approx(0.04 + 0.02j)


def test_iq_inverse():
    from satlab.modem import qpsk

    p = qpsk(np.tile([0, 0, 0, 1, 1, 0, 1, 1], 10))
    rx = m.iq_imbalance(p, 0.7, 0.3, 0.04j)
    fit = m.fit_iq(rx, p)
    assert np.allclose(m.correct_iq(rx, fit["coefficients"]), p)


def test_amplifier_preserves_phase_and_limits_envelope():
    x = np.array([0.001, 1, 10]) * np.exp(0.4j)
    r = m.rapp(x, 0.5)
    assert np.allclose(np.angle(r), 0.4)
    assert max(abs(r)) <= 0.5
    assert abs(r[0] / x[0] - 1) < 1e-8


def test_adc_rails():
    r = m.quantize(np.array([-2 - 2j, 0, 2 + 2j]), 8, 1)
    assert r["samples"][0] == pytest.approx(-1 - 1j)
    assert r["samples"][-1] == pytest.approx(1 + 1j)
    assert np.array_equal(r["clipped"], [True, False, True])


def test_phase_increment_variance():
    r = m.phase_noise(np.ones(100000), 80000, 5, np.random.default_rng(5))
    assert np.var(np.diff(r["phase_rad"])) == pytest.approx(2 * np.pi * 5 / 80000, rel=0.02)
    assert np.allclose(abs(r["samples"]), 1)
