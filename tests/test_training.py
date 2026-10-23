"""Offline CPU checks for the optional training backend; no pretrained model download."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments  # noqa: E402

from sentiment_ensemble.training import TweetDataset, WeightedTrainer, binary_examples  # noqa: E402


def test_binary_boundaries_keep_the_original_subsets():
    rows = [
        {"ID": str(i), "TEXT": "example", "TARGET": label}
        for i, label in enumerate(["P", "N", "NEU", "NONE"])
    ]
    for task, expected_rows, expected_labels in [
        ("none", 4, [0, 0, 0, 1]),
        ("neu", 3, [0, 0, 1]),
        ("pn", 2, [1, 0]),
    ]:
        subset, labels = binary_examples(rows, task)
        assert len(subset) == expected_rows
        assert labels.tolist() == expected_labels


def test_weighted_training_and_prediction_on_tiny_local_model(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=24,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=4,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
    )
    dataset = TweetDataset(
        {"input_ids": [[1, 3, 2]] * 4, "attention_mask": [[1, 1, 1]] * 4}, np.array([0, 1, 2, 3])
    )
    trainer = WeightedTrainer(
        model=model,
        class_weights=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        smoothing=0.1,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            max_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
        ),
        train_dataset=dataset,
    )
    batch = {"input_ids": torch.tensor([[1, 3, 2], [1, 3, 2]]), "labels": torch.tensor([0, 1])}
    loss, outputs = trainer.compute_loss(model, batch, return_outputs=True)
    expected = torch.nn.functional.cross_entropy(
        outputs.logits, batch["labels"], weight=trainer.class_weights, label_smoothing=0.1
    )
    assert torch.allclose(loss, expected)
    assert trainer.train().global_step == 1
    assert trainer.predict(dataset).predictions.shape == (4, 4)


def test_training_exports_provenance_and_refuses_existing_outputs(tmp_path, monkeypatch):
    import hashlib
    import json

    import sentiment_ensemble.training as backend

    train = tmp_path / "train.csv"
    train.write_text("ID,TEXT,TARGET\n1,a,P\n2,b,N\n3,c,NEU\n4,d,NONE\n", encoding="utf-8")
    test = tmp_path / "test.csv"
    test.write_text("ID,TEXT\n5,new\n6,post\n", encoding="utf-8")
    expected = np.array([[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]], dtype=np.float32)

    def fit(recipe, records, test_records, labels, destination, **kwargs):
        assert labels.tolist() == [0, 1, 2, 3]
        assert len(test_records) == 2
        assert kwargs["seed"] == 42
        return expected

    monkeypatch.setattr(backend, "fit_classifier", fit)
    output_dir = tmp_path / "output"
    output = backend.train_component("rob42", train, test, output_dir)
    np.testing.assert_array_equal(np.load(output), expected)
    metadata = json.loads((output_dir / "rob42/training.json").read_text(encoding="utf-8"))
    assert metadata["input_sha256"]["train"] == hashlib.sha256(train.read_bytes()).hexdigest()
    assert metadata["class_order"] == ["P", "N", "NEU", "NONE"]
    with pytest.raises(ValueError, match="already exist"):
        backend.train_component("rob42", train, test, output_dir)
