import numpy as np
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


def test_handover_ignores_brief_score_chatter_and_switches_after_persistent_advantage():
    r = m.Handover(minimum_dwell_s=10, confirmation_s=3)
    assert r.update(0, {"a": 10, "b": 9}) == "a"
    assert r.update(8, {"a": 10, "b": 14}) == "a"
    assert r.update(9, {"a": 10, "b": 9}) == "a"
    assert r.update(10, {"a": 10, "b": 14}) == "a"
    assert r.update(13, {"a": 10, "b": 14}) == "b"
    assert r.handovers == 1


def test_service_integral_does_not_credit_missing_blocks_or_zero_padding():
    r = m.accumulate([0, 1, 3], [10, 20, 100], [10, 0, 0])
    assert np.allclose(r["offered_bits"], [0, 10, 50])
    assert np.allclose(r["delivered_bits"], [0, 10, 10])
