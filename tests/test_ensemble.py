"""Formula parity, malformed inputs, stable softmax and output contracts."""

from pathlib import Path

import numpy as np
import pytest

from sentiment_ensemble.ensemble import FILES, LABELS, compose_cascade, predict, probabilities


def test_label_order_and_strong_class_decisions():
    logits = {key: np.eye(4) * 8 for key in FILES}
    assert LABELS == ("P", "N", "NEU", "NONE")
    np.testing.assert_array_equal(predict(logits), [0, 1, 2, 3])


def test_boost_breaks_equal_logit_tie_in_historical_class_order():
    logits = {key: np.zeros((1, 4)) for key in FILES}
    assert predict(logits).tolist() == [2]


def test_softmax_stays_finite_with_large_logits():
    logits = {key: np.array([[10000.0, -10000.0, 0.0, 1.0]]) for key in FILES}
    result = probabilities(logits)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result.sum(axis=1), 1)


@pytest.mark.parametrize(
    "bad", [np.zeros((2, 3)), np.zeros((0, 4)), np.full((2, 4), np.nan), np.full((2, 4), np.inf)]
)
def test_invalid_matrices_are_rejected(bad):
    logits = {key: np.zeros((2, 4)) for key in FILES}
    logits["beto"] = bad
    with pytest.raises(ValueError):
        predict(logits)


def test_row_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="rows"):
        predict({key: np.zeros((2, 4)) for key in FILES}, expected_rows=3)


def test_missing_component_is_rejected():
    with pytest.raises(ValueError, match="Missing"):
        predict({})


def test_predictions_match_original_synthetic_fixture():
    with np.load(Path(__file__).parent / "fixtures/ensemble.npz", allow_pickle=False) as fixture:
        logits = {key: fixture[key] for key in FILES}
        np.testing.assert_array_equal(predict(logits), fixture["expected"])


def test_cascade_composition_preserves_class_order():
    logits = compose_cascade(np.array([0.2]), np.array([0.25]), np.array([0.75]))
    np.testing.assert_allclose(np.exp(logits), [[0.45, 0.15, 0.20, 0.20]], rtol=1e-6)


def test_cascade_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="probabilities"):
        compose_cascade(np.array([1.2]), np.array([0.2]), np.array([0.5]))
