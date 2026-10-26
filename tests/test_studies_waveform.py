import numpy as np

from satlab import studies_waveform as m


def test_nominal():
    r = m.nominal()
    assert 0 <= r["summary"]["ber"] <= 1
    assert np.isfinite(r["arrays"]["recovered_symbols"]).all()
