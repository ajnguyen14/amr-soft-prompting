"""Smoke tests for AMRLoss and SingleTargetLoss (src/training/loss.py)."""

import pytest
import torch

from src.training.loss import AMRLoss, SingleTargetLoss

NUM_FAMILIES = 4
NUM_MECHANISMS = 5
NUM_DRUG_CLASSES = 6
BATCH_SIZE = 3


# ---------------------------------------------------------------------------
# AMRLoss (V1)
# ---------------------------------------------------------------------------


class TestAMRLoss:
    def test_returns_expected_keys(self):
        loss_fn = AMRLoss()
        logits = {"amr_gene_family": torch.randn(BATCH_SIZE, NUM_FAMILIES)}
        batch = {"amr_gene_family": torch.randint(0, NUM_FAMILIES, (BATCH_SIZE,))}
        losses = loss_fn(logits, batch)
        assert set(losses.keys()) == {"amr_gene_family", "total"}
        assert losses["total"].dim() == 0

    def test_weight_scales_total(self):
        logits = {"amr_gene_family": torch.randn(BATCH_SIZE, NUM_FAMILIES)}
        batch = {"amr_gene_family": torch.randint(0, NUM_FAMILIES, (BATCH_SIZE,))}
        unweighted = AMRLoss(weight_amr_gene_family=1.0)(logits, batch)
        weighted = AMRLoss(weight_amr_gene_family=2.0)(logits, batch)
        assert torch.isclose(weighted["total"], 2.0 * unweighted["total"])


# ---------------------------------------------------------------------------
# SingleTargetLoss (V2 -- Runs 1-3)
# ---------------------------------------------------------------------------


class TestSingleTargetLossCrossEntropy:
    """loss_type='ce', matching Run 2 (resistance_mechanism) / Run 3 (amr_gene_family)."""

    def test_returns_expected_keys(self):
        loss_fn = SingleTargetLoss(
            target_name="resistance_mechanism", batch_key="resistance_mechanism", loss_type="ce"
        )
        logits = {"resistance_mechanism": torch.randn(BATCH_SIZE, NUM_MECHANISMS)}
        batch = {"resistance_mechanism": torch.randint(0, NUM_MECHANISMS, (BATCH_SIZE,))}
        losses = loss_fn(logits, batch)
        assert set(losses.keys()) == {"resistance_mechanism", "total"}
        assert losses["total"].dim() == 0

    def test_uses_cross_entropy(self):
        loss_fn = SingleTargetLoss(
            target_name="resistance_mechanism", batch_key="resistance_mechanism", loss_type="ce"
        )
        assert isinstance(loss_fn.loss_fn, torch.nn.CrossEntropyLoss)


class TestSingleTargetLossBCE:
    """loss_type='bce', matching Run 1 (drug_class, multi-label).

    batch_key deliberately differs from target_name here (AMRDataset's
    multi-hot drug_class vector lives under 'drug_class_labels') -- this is
    the exact mismatch that would KeyError if the loss read batch[target_name]
    directly instead of batch[batch_key].
    """

    def test_returns_expected_keys_with_differing_batch_key(self):
        loss_fn = SingleTargetLoss(
            target_name="drug_class", batch_key="drug_class_labels", loss_type="bce"
        )
        logits = {"drug_class": torch.randn(BATCH_SIZE, NUM_DRUG_CLASSES)}
        batch = {"drug_class_labels": (torch.rand(BATCH_SIZE, NUM_DRUG_CLASSES) > 0.7).float()}
        losses = loss_fn(logits, batch)
        assert set(losses.keys()) == {"drug_class", "total"}
        assert losses["total"].dim() == 0

    def test_uses_bce_with_logits(self):
        loss_fn = SingleTargetLoss(
            target_name="drug_class", batch_key="drug_class_labels", loss_type="bce"
        )
        assert isinstance(loss_fn.loss_fn, torch.nn.BCEWithLogitsLoss)


class TestSingleTargetLossInvalidType:
    def test_unsupported_loss_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            SingleTargetLoss(target_name="drug_class", batch_key="drug_class_labels", loss_type="mse")
