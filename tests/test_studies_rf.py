from satlab import studies_rf as m


def test_ber():
    r = m.ber()
    assert r["summary"]["max_absolute_error"] < 0.003


def test_burst():
    r = m.burst()
    assert r["summary"]["ber"] < 0.01
