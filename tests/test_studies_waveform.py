import numpy as np
import pytest

from satlab import studies_waveform as m


def test_nominal():
    r = m.nominal()
    assert 0 <= r["summary"]["ber"] <= 1
    assert np.isfinite(r["arrays"]["recovered_symbols"]).all()


def test_uncalibrated():
    r = m.uncalibrated()
    assert 0 <= r["summary"]["ber"] <= 1
    assert np.isfinite(r["arrays"]["recovered_symbols"]).all()


def test_timing():
    r = m.timing()
    assert 0 <= r["summary"]["ber"] <= 1
    assert np.isfinite(r["arrays"]["recovered_symbols"]).all()


def test_iq():
    r = m.iq()
    assert 0 <= r["summary"]["ber"] <= 1
    assert np.isfinite(r["arrays"]["recovered_symbols"]).all()


def test_clipping():
    r = m.clipping()
    assert 0 <= r["summary"]["ber"] <= 1
    assert np.isfinite(r["arrays"]["recovered_symbols"]).all()


def test_amplifier():
    r = m.amplifier()
    assert 0 <= r["summary"]["ber"] <= 1
    assert np.isfinite(r["arrays"]["recovered_symbols"]).all()


def test_phase():
    r = m.phase()
    assert 0 <= r["summary"]["ber"] <= 1
    assert np.isfinite(r["arrays"]["recovered_symbols"]).all()


def test_clock():
    r = m.clock()
    assert 0 <= r["summary"]["ber"] <= 1
    assert np.isfinite(r["arrays"]["recovered_symbols"]).all()


def test_nyquist():
    assert m.nyquist()["summary"]["maximum_off_center_isi"] < 0.02


def test_image_rejection():
    s = m.image_rejection()["summary"]
    assert s["image_rejection_after_db"] > s["image_rejection_before_db"] + 60


def test_noise():
    assert m.noise()["summary"]["observed_variance"] == pytest.approx(0.1, rel=0.06)


def test_carrier_limit():
    s = m.carrier_limit()["summary"]
    assert s["maximum_inside_error_hz"] < 0.01 and s["maximum_outside_error_hz"] > 50
