"""Explicit training recipes for the four components retained in the ensemble."""

from __future__ import annotations

from dataclasses import dataclass

from .preprocessing import CleanMode


@dataclass(frozen=True)
class TrainingRecipe:
    model: str
    clean_mode: CleanMode
    epochs: int = 4
    learning_rate: float = 2e-5
    max_length: int = 128
    label_smoothing: float = 0.10
    batch_size: int = 8
    gradient_accumulation: int = 2
    eval_batch_size: int = 16
    gradient_checkpointing: bool = False


RECIPES = {
    "rob42": TrainingRecipe("pysentimiento/robertuito-base-uncased", "robertuito_uncased"),
    "cardiff": TrainingRecipe(
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "cased",
        batch_size=4,
        gradient_accumulation=4,
        eval_batch_size=8,
        gradient_checkpointing=True,
    ),
    "beto": TrainingRecipe("dccuchile/bert-base-spanish-wwm-cased", "cased"),
    "cascade": TrainingRecipe(
        "pysentimiento/robertuito-sentiment-analysis", "robertuito_uncased", label_smoothing=0.05
    ),
}
CASCADE_EPOCHS = {"none": 4, "neu": 5, "pn": 3}
