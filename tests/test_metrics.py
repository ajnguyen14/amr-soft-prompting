"""Smoke tests for the shared metrics helper (src/eval/metrics.py)."""

import torch

from src.eval.metrics import compute_metrics

NUM_MECHANISMS = 3
NUM_FAMILIES = 4
NUM_DRUG_CLASSES = 5


def _one_hot_logits(indices: torch.Tensor, num_classes: int, magnitude: float = 10.0) -> torch.Tensor:
    """Build logits whose argmax exactly matches `indices` (confident predictions)."""
    logits = torch.full((len(indices), num_classes), -magnitude)
    logits[torch.arange(len(indices)), indices] = magnitude
    return logits


class TestAccuracyMetrics:
    def test_perfect_predictions_give_accuracy_one(self):
        mechanism_labels = torch.tensor([0, 1, 2])
        family_labels = torch.tensor([3, 2, 1])
        logits = {
            "resistance_mechanism": _one_hot_logits(mechanism_labels, NUM_MECHANISMS),
            "amr_gene_family": _one_hot_logits(family_labels, NUM_FAMILIES),
            "drug_class": torch.zeros(3, NUM_DRUG_CLASSES),
        }
        batch = {
            "resistance_mechanism": mechanism_labels,
            "amr_gene_family": family_labels,
            "drug_class_labels": torch.zeros(3, NUM_DRUG_CLASSES),
        }
        metrics = compute_metrics(logits, batch)
        assert metrics["resistance_mechanism_accuracy"] == 1.0
        assert metrics["amr_gene_family_accuracy"] == 1.0

    def test_all_wrong_predictions_give_accuracy_zero(self):
        mechanism_labels = torch.tensor([0, 1])
        wrong_pred = torch.tensor([1, 0])  # deliberately swapped
        logits = {
            "resistance_mechanism": _one_hot_logits(wrong_pred, NUM_MECHANISMS),
            "amr_gene_family": _one_hot_logits(wrong_pred, NUM_FAMILIES),
            "drug_class": torch.zeros(2, NUM_DRUG_CLASSES),
        }
        batch = {
            "resistance_mechanism": mechanism_labels,
            "amr_gene_family": mechanism_labels,
            "drug_class_labels": torch.zeros(2, NUM_DRUG_CLASSES),
        }
        metrics = compute_metrics(logits, batch)
        assert metrics["resistance_mechanism_accuracy"] == 0.0
        assert metrics["amr_gene_family_accuracy"] == 0.0


class TestDrugClassF1:
    def test_perfect_multi_label_predictions_give_f1_one(self):
        labels = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0]])
        # Large positive logit where label=1, large negative where label=0.
        logits = torch.where(labels == 1.0, torch.tensor(10.0), torch.tensor(-10.0))
        batch = {
            "drug_class_labels": labels,
            "resistance_mechanism": torch.tensor([0, 0]),
            "amr_gene_family": torch.tensor([0, 0]),
        }
        model_logits = {
            "drug_class": logits,
            "resistance_mechanism": _one_hot_logits(torch.tensor([0, 0]), NUM_MECHANISMS),
            "amr_gene_family": _one_hot_logits(torch.tensor([0, 0]), NUM_FAMILIES),
        }
        metrics = compute_metrics(model_logits, batch)
        assert metrics["drug_class_f1_micro"] == 1.0

    def test_f1_is_bounded_between_zero_and_one(self):
        torch.manual_seed(0)
        labels = (torch.rand(4, NUM_DRUG_CLASSES) > 0.5).float()
        logits = torch.randn(4, NUM_DRUG_CLASSES)
        batch = {
            "drug_class_labels": labels,
            "resistance_mechanism": torch.tensor([0, 1, 0, 1]),
            "amr_gene_family": torch.tensor([0, 1, 2, 3]),
        }
        model_logits = {
            "drug_class": logits,
            "resistance_mechanism": _one_hot_logits(torch.tensor([0, 1, 0, 1]), NUM_MECHANISMS),
            "amr_gene_family": _one_hot_logits(torch.tensor([0, 1, 2, 3]), NUM_FAMILIES),
        }
        metrics = compute_metrics(model_logits, batch)
        assert 0.0 <= metrics["drug_class_f1_micro"] <= 1.0


class TestReturnShape:
    def test_returns_expected_keys(self):
        batch_size = 2
        batch = {
            "drug_class_labels": torch.zeros(batch_size, NUM_DRUG_CLASSES),
            "resistance_mechanism": torch.zeros(batch_size, dtype=torch.long),
            "amr_gene_family": torch.zeros(batch_size, dtype=torch.long),
        }
        logits = {
            "drug_class": torch.zeros(batch_size, NUM_DRUG_CLASSES),
            "resistance_mechanism": torch.zeros(batch_size, NUM_MECHANISMS),
            "amr_gene_family": torch.zeros(batch_size, NUM_FAMILIES),
        }
        metrics = compute_metrics(logits, batch)
        assert set(metrics.keys()) == {
            "resistance_mechanism_accuracy",
            "amr_gene_family_accuracy",
            "drug_class_f1_micro",
        }
        assert all(isinstance(v, float) for v in metrics.values())
