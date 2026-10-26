import json

import pytest

from satlab.batch import capture, load_catalog
from satlab.experiments import ExperimentConfig


def test_data_batch_roundtrips_sealed_metadata_without_plots(tmp_path):
    capture([ExperimentConfig("coherent-frequency-loss")], tmp_path, data_only=True)
    result = load_catalog(tmp_path)[0]
    assert result["name"] == "coherent-frequency-loss"
    assert not (tmp_path / result["name"] / "figure.png").exists()
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    catalog["experiments"][0]["archive"] = "../outside.npz"
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="canonical"):
        load_catalog(tmp_path)


def test_batch_rejects_duplicate_output_names_before_writing(tmp_path):
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="duplicate"):
        capture([config, config], tmp_path)
    assert not list(tmp_path.iterdir())


def test_catalog_render_replays_original_source_and_detects_damaged_figure(tmp_path):
    from satlab.batch import render, verify

    capture([ExperimentConfig("coherent-frequency-loss")], tmp_path, data_only=True)
    source = load_catalog(tmp_path)[0]["provenance"]
    assert render(tmp_path) == 1
    assert verify(tmp_path) == 1
    assert load_catalog(tmp_path)[0]["provenance"] == source
    (tmp_path / "coherent-frequency-loss" / "figure.png").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="integrity"):
        verify(tmp_path)
