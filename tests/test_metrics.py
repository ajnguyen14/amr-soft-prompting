"""Smoke tests for the shared metrics helper (src/eval/metrics.py)."""

import pytest
import torch

from src.eval.metrics import compute_metrics, compute_single_target_metrics

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


# ---------------------------------------------------------------------------
# compute_single_target_metrics (V2 -- Runs 1-3)
# ---------------------------------------------------------------------------

NUM_MECHANISMS = 5
NUM_DRUG_CLASSES = 6


class TestSingleTargetCrossEntropy:
    """loss_type='ce', matching Run 2/3."""

    def test_perfect_predictions_give_accuracy_one(self):
        labels = torch.tensor([3, 1, 0])
        logits = {"resistance_mechanism": _one_hot_logits(labels, NUM_MECHANISMS)}
        batch = {"resistance_mechanism": labels}
        metrics = compute_single_target_metrics(
            "resistance_mechanism", "resistance_mechanism", "ce", logits, batch
        )
        assert metrics == {"resistance_mechanism_accuracy": 1.0}

    def test_all_wrong_predictions_give_accuracy_zero(self):
        labels = torch.tensor([0, 1])
        wrong_pred = torch.tensor([1, 0])
        logits = {"resistance_mechanism": _one_hot_logits(wrong_pred, NUM_MECHANISMS)}
        batch = {"resistance_mechanism": labels}
        metrics = compute_single_target_metrics(
            "resistance_mechanism", "resistance_mechanism", "ce", logits, batch
        )
        assert metrics["resistance_mechanism_accuracy"] == 0.0


class TestSingleTargetBCE:
    """loss_type='bce', matching Run 1 (drug_class). batch_key deliberately
    differs from target_name, mirroring AMRDataset's 'drug_class_labels' key.
    """

    def test_perfect_predictions_give_subset_accuracy_and_f1_one(self):
        labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        # Large-magnitude logits with the same sign as the label so
        # sigmoid(...) > 0.5 threshold reproduces `labels` exactly.
        logits = {"drug_class": torch.where(labels == 1.0, 10.0, -10.0)}
        batch = {"drug_class_labels": labels}
        metrics = compute_single_target_metrics(
            "drug_class", "drug_class_labels", "bce", logits, batch
        )
        assert metrics["drug_class_subset_accuracy"] == 1.0
        assert metrics["drug_class_micro_f1"] == 1.0

    def test_all_wrong_predictions_give_zero_f1(self):
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        wrong_pred = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        logits = {"drug_class": torch.where(wrong_pred == 1.0, 10.0, -10.0)}
        batch = {"drug_class_labels": labels}
        metrics = compute_single_target_metrics(
            "drug_class", "drug_class_labels", "bce", logits, batch
        )
        assert metrics["drug_class_subset_accuracy"] == 0.0
        assert metrics["drug_class_micro_f1"] == 0.0

    def test_returns_expected_keys(self):
        batch_size = 2
        batch = {"drug_class_labels": torch.zeros(batch_size, NUM_DRUG_CLASSES)}
        logits = {"drug_class": torch.full((batch_size, NUM_DRUG_CLASSES), -10.0)}
        metrics = compute_single_target_metrics(
            "drug_class", "drug_class_labels", "bce", logits, batch
        )
        assert set(metrics.keys()) == {"drug_class_subset_accuracy", "drug_class_micro_f1"}
        assert all(isinstance(v, float) for v in metrics.values())


class TestSingleTargetInvalidLossType:
    def test_unsupported_loss_type_raises(self):
        batch = {"amr_gene_family": torch.zeros(2, dtype=torch.long)}
        logits = {"amr_gene_family": torch.zeros(2, NUM_FAMILIES)}
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            compute_single_target_metrics(
                "amr_gene_family", "amr_gene_family", "mse", logits, batch
            )
