from satlab import studies_orbit as m


def test_invariants():
    r = m.invariants()
    assert r["summary"]["max_energy_relative"] < 1e-10
    assert r["summary"]["max_h_relative"] < 1e-10


def test_convergence():
    r = m.convergence()
    assert 3.8 < r["summary"]["observed_order"] < 4.4


def test_precession():
    r = m.precession()
    s = r["summary"]
    assert abs(s["measured_raan_rad_s"] / s["secular_raan_rad_s"] - 1) < 0.06
    assert s["max_hz_relative"] < 1e-7


def test_doppler():
    r = m.doppler()
    assert r["summary"]["interior_max_error_hz"] < 2
