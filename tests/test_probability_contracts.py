"""Numerical and batching invariants for replay without pretrained models or data."""

import numpy as np
import pytest

from sentiment_ensemble.ensemble import FILES, compose_cascade, predict, probabilities


def sample_logits():
    rng = np.random.default_rng(20261024)
    return {component: rng.normal(size=(17, 4)) for component in FILES}


def test_rowwise_logit_offsets_do_not_change_probabilities():
    logits = sample_logits()
    offsets = np.arange(17, dtype=float)[:, None] * 100
    shifted = {component: values + offsets for component, values in logits.items()}
    np.testing.assert_allclose(probabilities(shifted), probabilities(logits), atol=1e-12)


def test_batch_partitioning_preserves_scores_and_predictions():
    logits = sample_logits()
    pieces = [
        probabilities({key: values[start:end] for key, values in logits.items()})
        for start, end in [(0, 1), (1, 8), (8, 17)]
    ]
    combined = np.concatenate(pieces)
    np.testing.assert_allclose(combined, probabilities(logits), atol=1e-14)
    np.testing.assert_array_equal(combined.argmax(axis=1), predict(logits))


def test_row_permutation_preserves_corresponding_predictions():
    logits = sample_logits()
    permutation = np.random.default_rng(41).permutation(17)
    permuted = {key: values[permutation] for key, values in logits.items()}
    np.testing.assert_array_equal(predict(permuted), predict(logits)[permutation])


def test_replay_does_not_mutate_or_require_writable_inputs():
    logits = sample_logits()
    originals = {key: values.copy() for key, values in logits.items()}
    for values in logits.values():
        values.setflags(write=False)
    scores = probabilities(logits)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()
    np.testing.assert_allclose(scores.sum(axis=1), 1)
    for key in logits:
        np.testing.assert_array_equal(logits[key], originals[key])


def test_cascade_boundary_probabilities_produce_expected_classes():
    # Certain NONE, NEU, positive and negative branches, in that order.
    scores = compose_cascade(np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]))
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(scores.argmax(axis=1), [3, 2, 0, 1])
    np.testing.assert_allclose(np.exp(scores).sum(axis=1), 1, atol=1e-6)


@pytest.mark.parametrize("bad", [np.array([]), np.zeros((2, 1)), np.array([np.nan])])
def test_cascade_rejects_empty_nonvector_or_nonfinite_inputs(bad):
    with pytest.raises(ValueError):
        compose_cascade(bad, bad, bad)


def test_cascade_rejects_misaligned_vectors():
    with pytest.raises(ValueError, match="matching shapes"):
        compose_cascade(np.array([0.1, 0.2]), np.array([0.3]), np.array([0.4]))
