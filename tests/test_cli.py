import numpy as np
import pytest

from sentiment_ensemble.cli import main
from sentiment_ensemble.ensemble import FILES


def inputs(tmp_path):
    test = tmp_path / "test.csv"
    test.write_text("ID\n001\n002\n", encoding="utf-8")
    for filename in FILES.values():
        np.save(tmp_path / filename, np.array([[9.0, 0.0, 0.0, 0.0], [0.0, 9.0, 0.0, 0.0]]))
    return ["--logits-dir", str(tmp_path), "--test-csv", str(test)]


def test_replay_preserves_ids_and_does_not_overwrite(tmp_path):
    args = inputs(tmp_path)
    out = tmp_path / "output.csv"
    assert main(["replay", *args, "--output", str(out)]) == 0
    assert out.read_bytes() == b"ID,TARGET\r\n001,P\r\n002,N\r\n"
    before = out.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        main(["replay", *args, "--output", str(out)])
    assert before == out.read_bytes()


def test_reference_mismatch_does_not_write_submission(tmp_path):
    args = inputs(tmp_path)
    ref = tmp_path / "wrong.csv"
    ref.write_text("ID,TARGET\n001,N\n002,P\n", encoding="utf-8")
    out = tmp_path / "output.csv"
    with pytest.raises(SystemExit, match="2"):
        main(["replay", *args, "--output", str(out), "--reference", str(ref)])
    assert not out.exists()


def test_dry_run_does_not_create_training_outputs(tmp_path, capsys):
    import json

    train = tmp_path / "train.csv"
    train.write_text(
        "ID,TEXT,TARGET\n1,good,P\n2,bad,N\n3,neutral,NEU\n3,fact,NONE\n", encoding="utf-8"
    )
    test = tmp_path / "test.csv"
    test.write_text("ID,TEXT\n5,example\n", encoding="utf-8")
    out = tmp_path / "training"
    assert (
        main(
            [
                "train",
                "--component",
                "cascade",
                "--train-csv",
                str(train),
                "--test-csv",
                str(test),
                "--output-dir",
                str(out),
                "--dry-run",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["train_rows"] == 4
    assert not out.exists()
