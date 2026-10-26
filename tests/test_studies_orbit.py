from satlab import studies_orbit as m


def test_invariants():
    r = m.invariants()
    assert r["summary"]["max_energy_relative"] < 1e-10
    assert r["summary"]["max_h_relative"] < 1e-10
