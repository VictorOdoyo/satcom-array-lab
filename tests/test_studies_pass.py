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


def test_one_bit():
    r = m.one_bit()
    assert r["name"] == "one-bit-phase-pass"
    assert r["summary"]["delivered_mbit"] >= 0
    assert np.all(r["arrays"]["delivered_bits"] <= r["arrays"]["offered_bits"])


def test_calibration():
    r = m.calibration()
    assert r["name"] == "calibration-error-pass"
    assert r["summary"]["delivered_mbit"] >= 0
    assert np.all(r["arrays"]["delivered_bits"] <= r["arrays"]["offered_bits"])


def test_failures():
    r = m.failures()
    assert r["name"] == "element-failure-pass"
    assert r["summary"]["delivered_mbit"] >= 0
    assert np.all(r["arrays"]["delivered_bits"] <= r["arrays"]["offered_bits"])
