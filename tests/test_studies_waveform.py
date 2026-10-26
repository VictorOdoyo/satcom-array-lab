import numpy as np

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
