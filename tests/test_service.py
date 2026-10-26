import pytest

from satlab import service as m


def test_service_rates_keep_coding_and_framing_overhead_explicit():
    assert m.information_rate(m.MODES[0], 2e6) == pytest.approx(1.9e6)
    assert m.information_rate(m.MODES[2], 2e6) == pytest.approx(5.7e6)


def test_adaptive_mode_downgrades_immediately_and_guards_upgrades():
    r = m.AdaptiveMode(dwell_s=5)
    assert r.update(0, 4) == m.MODES[0]
    assert r.update(1, 12) == m.MODES[0]
    assert r.update(6, 12) == m.MODES[2]
    assert r.update(7, 5) == m.MODES[0]
    assert r.update(8, -2) is None
