"""Train a base classifier or the three binary cascade boundaries.

Imported only by the train command; CPU replay does not load PyTorch or Transformers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import CASCADE_EPOCHS, RECIPES, TrainingRecipe
from .ensemble import FILES, LABELS, compose_cascade
from .io import read_records, write_submission
from .preprocessing import clean_text

logger = logging.getLogger(__name__)


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict[str, list], labels: np.ndarray | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, smoothing: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.smoothing = smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, model.config.num_labels),
            labels.view(-1),
            weight=self.class_weights.to(outputs.logits.device),
            label_smoothing=self.smoothing,
        )
        return (loss, outputs) if return_outputs else loss


def binary_examples(
    records: list[dict[str, str]], task: str
) -> tuple[list[dict[str, str]], np.ndarray]:
    """Select the historical training subset and positive label for each boundary."""
    if task not in CASCADE_EPOCHS:
        raise ValueError(f"Unknown cascade task: {task}")
    allowed = {"none": set(LABELS), "neu": {"P", "N", "NEU"}, "pn": {"P", "N"}}[task]
    positive = {"none": "NONE", "neu": "NEU", "pn": "P"}[task]
    subset = [row for row in records if row["TARGET"] in allowed]
    return subset, np.asarray([int(row["TARGET"] == positive) for row in subset])


def fit_classifier(
    recipe: TrainingRecipe,
    records: list[dict[str, str]],
    test: list[dict[str, str]],
    labels: np.ndarray,
    destination: Path,
    *,
    epochs: int,
    seed: int,
    tokenizer=None,
) -> np.ndarray:
    """Fit one classification head and return test logits without changing row order."""
    classes = np.unique(labels)
    if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Training requires every class with contiguous zero-based labels")
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(recipe.model, use_fast=True)
    label_names = list(LABELS) if len(classes) == 4 else ["NO", "YES"]
    model = AutoModelForSequenceClassification.from_pretrained(
        recipe.model,
        num_labels=len(classes),
        id2label=dict(enumerate(label_names)),
        label2id={label: index for index, label in enumerate(label_names)},
        ignore_mismatched_sizes=True,
    )
    if recipe.gradient_checkpointing:
        model.config.use_cache = False

    def encode(rows):
        return tokenizer(
            [clean_text(row["TEXT"], recipe.clean_mode) for row in rows],
            truncation=True,
            padding=True,
            max_length=recipe.max_length,
        )

    weights = torch.tensor(
        compute_class_weight(class_weight="balanced", classes=classes, y=labels),
        dtype=torch.float32,
    )
    args = TrainingArguments(
        output_dir=str(destination / "logs"),
        num_train_epochs=epochs,
        learning_rate=recipe.learning_rate,
        per_device_train_batch_size=recipe.batch_size,
        per_device_eval_batch_size=recipe.eval_batch_size,
        gradient_accumulation_steps=recipe.gradient_accumulation,
        warmup_ratio=0.10,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="no",
        eval_strategy="no",
        seed=seed,
        dataloader_num_workers=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=recipe.gradient_checkpointing,
        remove_unused_columns=False,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=TweetDataset(encode(records), labels),
        class_weights=weights,
        smoothing=recipe.label_smoothing,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(destination / "model"))
    tokenizer.save_pretrained(destination / "model")
    return trainer.predict(TweetDataset(encode(test))).predictions


def train_component(
    component: str, train_csv: Path, test_csv: Path, output_dir: Path, *, seed: int = 42
) -> Path:
    """Train into a new component directory; reference logits are never overwritten."""
    recipe = RECIPES[component]
    train = read_records(train_csv, ("ID", "TEXT", "TARGET"), unique_ids=False)
    test = read_records(test_csv, ("ID", "TEXT"))
    if set(row["TARGET"] for row in train) != set(LABELS):
        raise ValueError("Training CSV must contain all four P/N/NEU/NONE classes")
    destination = output_dir / component
    output = output_dir / FILES[component]
    if output.exists() or destination.exists():
        raise ValueError(f"Training outputs already exist for {component}; choose a new directory")
    destination.mkdir(parents=True)
    for variable in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ.setdefault(variable, "8")
    torch.set_num_threads(8)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Training %s on %s", component, "CUDA" if torch.cuda.is_available() else "CPU")
    if component == "cascade":
        tokenizer = AutoTokenizer.from_pretrained(recipe.model, use_fast=True)
        probabilities = []
        for task, epochs in CASCADE_EPOCHS.items():
            subset, labels = binary_examples(train, task)
            logits = fit_classifier(
                recipe,
                subset,
                test,
                labels,
                destination / task,
                epochs=epochs,
                seed=seed,
                tokenizer=tokenizer,
            ).astype(np.float64)
            exponent = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilities.append((exponent / exponent.sum(axis=1, keepdims=True))[:, 1])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        scores = compose_cascade(*probabilities)
    else:
        labels = np.asarray([LABELS.index(row["TARGET"]) for row in train])
        scores = fit_classifier(
            recipe, train, test, labels, destination, epochs=recipe.epochs, seed=seed
        ).astype(np.float32)
    np.save(output, scores, allow_pickle=False)
    write_submission(
        destination / "predictions.csv", [row["ID"] for row in test], scores.argmax(axis=1)
    )
    metadata = {
        "component": component,
        "recipe": asdict(recipe),
        "seed": seed,
        "class_order": list(LABELS),
        "rows": {"train": len(train), "test": len(test)},
        "input_sha256": {
            "train": hashlib.sha256(train_csv.read_bytes()).hexdigest(),
            "test": hashlib.sha256(test_csv.read_bytes()).hexdigest(),
        },
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "versions": {
            name: version(name) for name in ["numpy", "torch", "transformers", "scikit-learn"]
        },
    }
    (destination / "training.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output
