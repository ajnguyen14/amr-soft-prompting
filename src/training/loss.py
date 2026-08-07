"""amr_gene_family loss used to train the V1 model."""

import torch
import torch.nn as nn


class AMRLoss(nn.Module):
    """Weighted amr_gene_family loss used to train the V1 model.

    resistance_mechanism and drug_class are deliberately not scored here --
    both are fed into SoftPromptModule as conditioning input, so training the
    classifier to predict them back out would only teach it to decode their
    own soft-prompt embedding rather than learn from the ESM-2 sequence
    representation (see docs/STATUS.md's label-leakage note). amr_gene_family
    is single-label (CrossEntropyLoss).

    Args:
        weight_amr_gene_family: Weight applied to the amr_gene_family loss
            term. A constructor arg (rather than hardcoded 1.0) so a future
            run can tune it via config without editing this file.
    """

    def __init__(
        self,
        weight_amr_gene_family: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight_amr_gene_family = weight_amr_gene_family
        self.amr_gene_family_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute the amr_gene_family loss and its weighted total.

        Args:
            logits: Output of ClassifierHead.forward() -- dict with key
                'amr_gene_family'.
            batch: Batch dict as produced by AMRDataset's default collation --
                key 'amr_gene_family'.

        Returns:
            Dict with key 'amr_gene_family' (the unweighted loss, for
            individual logging) plus 'total' (the weighted loss to call
            .backward() on).
        """
        amr_gene_family_loss = self.amr_gene_family_loss_fn(
            logits["amr_gene_family"], batch["amr_gene_family"]
        )

        total = self.weight_amr_gene_family * amr_gene_family_loss

        return {
            "amr_gene_family": amr_gene_family_loss,
            "total": total,
        }


class SingleTargetLoss(nn.Module):
    """Weighted loss for one configurable V2 target.

    Dispatches to the loss function CLAUDE.md's Single-Head Architecture
    table specifies per run: 'bce' (BCEWithLogitsLoss, Run 1's multi-label
    drug_class target) or 'ce' (CrossEntropyLoss, Run 2/3's single-label
    targets). Kept as a separate class rather than generalizing AMRLoss in
    place, since AMRLoss's fixed CrossEntropyLoss-on-amr_gene_family is what
    the two trained V1 checkpoints were actually optimized against.

    Args:
        target_name: Name of the prediction target, used as the logits dict
            key (matches SingleTargetClassifierHead's output key).
        batch_key: Key the target's label actually lives under in an
            AMRDataset-collated batch. Usually equal to target_name, except
            drug_class, whose multi-hot label is 'drug_class_labels' -- see
            src.data.dataset.TARGET_FIELD_SPECS, the source of truth for this
            mapping.
        loss_type: 'bce' for BCEWithLogitsLoss (multi-label) or 'ce' for
            CrossEntropyLoss (single-label) -- also from TARGET_FIELD_SPECS.
        weight: Weight applied to this term's loss. A constructor arg (rather
            than hardcoded 1.0) so a future run can tune it via config
            without editing this file.

    Raises:
        ValueError: If loss_type isn't 'bce' or 'ce'.
    """

    def __init__(
        self,
        target_name: str,
        batch_key: str,
        loss_type: str,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.target_name = target_name
        self.batch_key = batch_key
        self.loss_type = loss_type
        self.weight = weight

        if loss_type == "bce":
            self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()
        elif loss_type == "ce":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss_type {loss_type!r} -- must be 'bce' or 'ce'")

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute the configured target's loss and its weighted total.

        Args:
            logits: Output of SingleTargetClassifierHead.forward() -- dict
                with key self.target_name.
            batch: Batch dict as produced by AMRDataset's default collation --
                dict with key self.batch_key (may differ from
                self.target_name, e.g. drug_class's multi-hot label lives
                under 'drug_class_labels').

        Returns:
            Dict with key self.target_name (the unweighted loss, for
            individual logging) plus 'total' (the weighted loss to call
            .backward() on).
        """
        target_loss = self.loss_fn(logits[self.target_name], batch[self.batch_key])
        total = self.weight * target_loss

        return {
            self.target_name: target_loss,
            "total": total,
        }
