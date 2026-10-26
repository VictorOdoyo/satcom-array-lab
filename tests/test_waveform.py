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
