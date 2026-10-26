import pytest

from satlab import passmodel as m


def test_config_validation():
    assert m.Config().to_dict()["carrier_hz"] == 20e9
    with pytest.raises(ValueError):
        m.Config(failed_fraction=2)
    with pytest.raises(ValueError):
        m.Config(beacon_gap_s=(5, 1))
