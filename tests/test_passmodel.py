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


def test_panel_coordinates():
    r = m.panel_rotation()
    assert np.allclose(r.T @ r, np.eye(3))
    assert np.linalg.det(r) == pytest.approx(1)
    assert m.command_direction([1, 0, 0], [1, 1])[2] > 0


def test_forecast_uses_nominal_geometry():
    from satlab.array import Aperture

    _, s = m.geometry(m.Config(duration_s=2))
    score = m.forecast(s[0]["nominal"][1], m.Config(), Aperture())
    assert 65 < score < 85
    assert m.forecast({**s[0]["nominal"][1], "visible": False}, m.Config(), Aperture()) is None
