from satlab import studies_rf as m


def test_ber():
    r = m.ber()
    assert r["summary"]["max_absolute_error"] < 0.003
