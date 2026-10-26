import numpy as np
import pytest

from satlab import passmodel as m


def test_config_validation():
    assert m.Config().to_dict()["carrier_hz"] == 20e9
    with pytest.raises(ValueError):
        m.Config(failed_fraction=2)
    with pytest.raises(ValueError):
        m.Config(beacon_gap_s=(5, 1))


def test_geometry_at_center():
    t, s = m.geometry(m.Config(duration_s=20, step_s=2))
    k = np.argmin(abs(t))
    assert s[0]["true"][k]["range_m"] == pytest.approx(600000, rel=1e-8)
    assert s[0]["nominal"][k]["range_m"] == pytest.approx(610000, rel=1e-8)
