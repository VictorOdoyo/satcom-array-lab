import pytest

from satlab import service as m


def test_service_rates_keep_coding_and_framing_overhead_explicit():
    assert m.information_rate(m.MODES[0], 2e6) == pytest.approx(1.9e6)
    assert m.information_rate(m.MODES[2], 2e6) == pytest.approx(5.7e6)
