"""Replay the selected InterTASS ensemble without loading neural model weights.

Input matrices must share row order and columns P, N, NEU, NONE. Casting to float64,
weight normalization, operation order and renormalization preserve the v22 formula.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

LABELS = ("P", "N", "NEU", "NONE")
FILES = {
    "rob42": "logits_seed42.npy",
    "cardiff": "logits_cardiff.npy",
    "beto": "logits_beto.npy",
    "cascade": "logits_cascade_robertuito_gpu.npy",
}
WEIGHTS = {"rob42": 0.40, "cardiff": 0.31, "beto": 0.19, "cascade": 0.10}
BOOST = (0.97, 1.00, 1.08, 1.08)


def probabilities(logits: Mapping[str, NDArray], expected_rows: int | None = None) -> NDArray:
    """Validate and combine four matrices; return adjusted, normalized scores.

    These are ensemble scores, not empirically calibrated confidence estimates.
    """
    combined = None
    rows = expected_rows
    total = float(sum(WEIGHTS.values()))
    for name, weight in WEIGHTS.items():
        if name not in logits:
            raise ValueError(f"Missing component: {name}")
        arr = np.asarray(logits[name], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 4 or arr.shape[0] == 0:
            raise ValueError(f"{name}: expected a non-empty (n, 4) matrix; got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name}: logits contain NaN or infinite values")
        if rows is None:
            rows = arr.shape[0]
        if arr.shape[0] != rows:
            raise ValueError(f"{name}: {arr.shape[0]} rows; expected {rows}")
        combined = (weight / total) * arr if combined is None else combined + (weight / total) * arr
    shifted = combined - np.max(combined, axis=1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp / np.sum(exp, axis=1, keepdims=True)
    adjusted = prob * np.asarray(BOOST, dtype=np.float64)[None, :]
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def predict(logits: Mapping[str, NDArray], expected_rows: int | None = None) -> NDArray:
    """Return integer labels in the fixed historical class order."""
    return probabilities(logits, expected_rows).argmax(axis=1)


def compose_cascade(p_none: NDArray, p_neu: NDArray, p_positive: NDArray) -> NDArray:
    """Compose binary probabilities in P/N/NEU/NONE order as historical float32 log scores."""
    arrays = [np.asarray(values, dtype=np.float64) for values in (p_none, p_neu, p_positive)]
    if any(values.ndim != 1 or values.size == 0 for values in arrays):
        raise ValueError("Cascade inputs must be non-empty vectors")
    if len({values.shape for values in arrays}) != 1:
        raise ValueError("Cascade inputs must have matching shapes")
    if any(
        not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
        for values in arrays
    ):
        raise ValueError("Cascade inputs must be probabilities between zero and one")
    none, neutral, positive = arrays
    scores = np.zeros((len(none), 4), dtype=np.float64)
    scores[:, 3] = none
    scores[:, 2] = (1.0 - none) * neutral
    scores[:, 0] = (1.0 - none) * (1.0 - neutral) * positive
    scores[:, 1] = (1.0 - none) * (1.0 - neutral) * (1.0 - positive)
    scores = np.clip(scores, 1e-12, 1.0)
    scores = scores / scores.sum(axis=1, keepdims=True)
    return np.log(scores).astype(np.float32)
