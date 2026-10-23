import pytest

from sentiment_ensemble.io import read_records


@pytest.mark.parametrize("content", ["ID\n1\n1\n", "ID\n \n", "TEXT\nhello\n", "ID,ID\n1,2\n"])
def test_invalid_prediction_ids_are_rejected(tmp_path, content):
    path = tmp_path / "test.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        read_records(path, ("ID",))


def test_training_can_preserve_repeated_ids(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("ID,TEXT,TARGET\n1,first,P\n1,second,N\n", encoding="utf-8")
    assert len(read_records(path, ("ID", "TEXT", "TARGET"), unique_ids=False)) == 2
