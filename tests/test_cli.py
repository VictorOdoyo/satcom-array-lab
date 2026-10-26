import pytest

from satlab.cli import main


def test_cli_lists_satellite_catalog(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "closed-loop-pass" in output and "constellation-handover" in output


def test_cli_runs_an_independent_coherent_study(tmp_path):
    from satlab.reporting import verify_report

    assert main(["run", "--case", "coherent-frequency-loss", "--output", str(tmp_path)]) == 0
    assert verify_report(tmp_path)


def test_cli_rejects_ambiguous_json_and_scale_overrides(tmp_path):
    path = tmp_path / "request.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--scale", "2", "--output", str(tmp_path)])
    assert error.value.code == 2


def test_cli_data_capture_render_and_replay_preserve_original_source(tmp_path):
    from satlab.reporting import load_archive

    source, replayed = tmp_path / "source", tmp_path / "replayed"
    main(["batch", "--cases", "coherent-frequency-loss", "--data-only", "--output", str(source)])
    main(["render", str(source)])
    main(["verify", str(source)])
    archive = source / "coherent-frequency-loss" / "fields.npz"
    provenance = load_archive(archive)["provenance"]
    main(["replay", str(archive), "--output", str(replayed)])
    assert load_archive(replayed / "fields.npz")["provenance"] == provenance
    main(["verify", str(replayed)])
    (replayed / "summary.csv").write_text("damaged")
    with pytest.raises(SystemExit) as error:
        main(["verify", str(replayed)])
    assert error.value.code == 2


def test_cli_help_exposes_supported_satellite_commands_only(capsys):
    from satlab.cli import parser

    text = parser().format_help()
    assert "satellite" in text
    assert "record-replay" not in text and "iq-audit" not in text
