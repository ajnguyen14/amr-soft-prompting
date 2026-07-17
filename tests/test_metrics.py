"""Smoke tests for the shared metrics helper (src/eval/metrics.py)."""

import torch

from src.eval.metrics import compute_metrics

NUM_FAMILIES = 4


def _one_hot_logits(indices: torch.Tensor, num_classes: int, magnitude: float = 10.0) -> torch.Tensor:
    """Build logits whose argmax exactly matches `indices` (confident predictions)."""
    logits = torch.full((len(indices), num_classes), -magnitude)
    logits[torch.arange(len(indices)), indices] = magnitude
    return logits


class TestAccuracyMetrics:
    def test_perfect_predictions_give_accuracy_one(self):
        family_labels = torch.tensor([3, 2, 1])
        logits = {"amr_gene_family": _one_hot_logits(family_labels, NUM_FAMILIES)}
        batch = {"amr_gene_family": family_labels}
        metrics = compute_metrics(logits, batch)
        assert metrics["amr_gene_family_accuracy"] == 1.0

    def test_all_wrong_predictions_give_accuracy_zero(self):
        family_labels = torch.tensor([0, 1])
        wrong_pred = torch.tensor([1, 0])  # deliberately swapped
        logits = {"amr_gene_family": _one_hot_logits(wrong_pred, NUM_FAMILIES)}
        batch = {"amr_gene_family": family_labels}
        metrics = compute_metrics(logits, batch)
        assert metrics["amr_gene_family_accuracy"] == 0.0


class TestReturnShape:
    def test_returns_expected_keys(self):
        batch_size = 2
        batch = {"amr_gene_family": torch.zeros(batch_size, dtype=torch.long)}
        logits = {"amr_gene_family": torch.zeros(batch_size, NUM_FAMILIES)}
        metrics = compute_metrics(logits, batch)
        assert set(metrics.keys()) == {"amr_gene_family_accuracy"}
        assert all(isinstance(v, float) for v in metrics.values())
