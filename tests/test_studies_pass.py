import numpy as np

from satlab import studies_pass as m


def test_nominal():
    r = m.nominal()
    assert r["name"] == "closed-loop-pass"
    assert r["summary"]["delivered_mbit"] >= 0
    assert np.all(r["arrays"]["delivered_bits"] <= r["arrays"]["offered_bits"])


def test_open_loop():
    r = m.open_loop()
    assert r["name"] == "ephemeris-only-pass"
    assert r["summary"]["delivered_mbit"] >= 0
    assert np.all(r["arrays"]["delivered_bits"] <= r["arrays"]["offered_bits"])


def test_rain():
    r = m.rain()
    assert r["name"] == "rain-fade-pass"
    assert r["summary"]["delivered_mbit"] >= 0
    assert np.all(r["arrays"]["delivered_bits"] <= r["arrays"]["offered_bits"])
