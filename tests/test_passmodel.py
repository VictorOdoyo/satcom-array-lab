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


def test_failed_aperture_unavailable():
    from satlab.array import Aperture

    _, s = m.geometry(m.Config(duration_s=2))
    p = Aperture()
    result = m.channel(
        s[0]["true"][1], np.array([0, 0, 1]), m.Config(), p, np.zeros(p.count), np.zeros(p.count, bool)
    )
    assert not result["available"]
    assert not np.any(result["weights"])


def test_causal_receiver_pipeline():
    result = m.simulate(m.Config(duration_s=80, step_s=1))
    assert result["delivered_bits"][-1] > 0
    assert result["offered_bit_s"][0] == 0
    assert np.all(result["delivered_bit_s"] <= result["offered_bit_s"])
    assert np.median(abs(result["residual_hz"])[30:]) < 30
    assert np.median(result["point_error_deg"][30:]) < 0.2


def test_raw_beacon_archive():
    r = m.simulate(m.Config(duration_s=30))
    assert r["true_state_eci"].shape == (31, 6)
    assert np.any(r["measured_carrier_hz"][r["beacon"].astype(bool)] != 0)
    assert np.linalg.norm(r["true_state_eci"][15, :3] - r["nominal_state_eci"][15, :3]) == pytest.approx(
        10000
    )


def test_beacon_gap_coast():
    r = m.simulate(m.Config(duration_s=80, beacon_gap_s=(-10, 10)))
    t = r["time_s"]
    middle = (t >= -4) & (t < 10)
    assert not np.any(r["beacon"][middle])
    assert not np.any(r["locked"][middle])
    assert not np.any(r["offered_bit_s"][middle])
    assert np.any(r["delivered_bit_s"][t > 15] > 0)


def test_open_loop_exposes_bias():
    closed = m.simulate(m.Config(duration_s=80))
    opened = m.simulate(m.Config(duration_s=80, closed_loop=False))
    assert closed["delivered_bits"][-1] > opened["delivered_bits"][-1]
    assert np.median(abs(closed["residual_hz"])[20:]) < np.median(abs(opened["residual_hz"])[20:]) / 10


def test_outlier_gating():
    r = m.simulate(m.Config(duration_s=80, inject_outlier=True))
    k = np.flatnonzero(r["time_s"] == 0)[0]
    assert r["beacon"][k] == 1
    assert r["accepted"][k] == 0
    assert r["nis"][k] > 9
    assert r["locked"][k + 1] == 1


def test_rain_degrades_service():
    clear = m.simulate(m.Config(duration_s=80))
    rain = m.simulate(m.Config(duration_s=80, rain_mm_h=20))
    assert np.mean(rain["cn0_dbhz"]) < np.mean(clear["cn0_dbhz"])
    assert rain["delivered_bits"][-1] < clear["delivered_bits"][-1]


def test_seed_reproduction():
    a = m.simulate(m.Config(duration_s=10, phase_std_rad=0.2, seed=22))
    b = m.simulate(m.Config(duration_s=10, phase_std_rad=0.2, seed=22))
    c = m.simulate(m.Config(duration_s=10, phase_std_rad=0.2, seed=23))
    assert np.array_equal(a["weights"], b["weights"])
    assert not np.array_equal(a["weights"], c["weights"])
