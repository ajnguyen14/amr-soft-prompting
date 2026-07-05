"""Combined multi-task loss for AMR gene classification."""

import torch
import torch.nn as nn


class AMRLoss(nn.Module):
    """Weighted sum of the three per-task losses used to train the V1 model.

    drug_class is multi-label (BCEWithLogitsLoss); resistance_mechanism and
    amr_gene_family are single-label (CrossEntropyLoss). Equal weighting
    (alpha = beta = gamma = 1) was settled during the 2026-07-02 design session,
    but all three weights are constructor args so a future run can tune them via
    config rather than editing this file.

    BCEWithLogitsLoss is summed then divided by batch_size (not its default
    'mean') so it lands on the same per-sample scale as CrossEntropyLoss's mean
    -- see the comment in __init__ for why the two losses' default reductions
    aren't comparable on their own.

    Args:
        weight_drug_class: Weight applied to the drug_class loss term.
        weight_resistance_mechanism: Weight applied to the resistance_mechanism
            loss term.
        weight_amr_gene_family: Weight applied to the amr_gene_family loss term.
    """

    # TODO: configs/base.yaml now has a `loss:` section (weight_drug_class,
    # weight_resistance_mechanism, weight_amr_gene_family) mirroring these
    # constructor defaults. Nothing reads it yet -- when train.py is written,
    # it must explicitly construct AMRLoss(**config["loss"]) rather than
    # relying on these defaults, or a change to the config file will silently
    # have no effect on the actual run.

    def __init__(
        self,
        weight_drug_class: float = 1.0,
        weight_resistance_mechanism: float = 1.0,
        weight_amr_gene_family: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight_drug_class = weight_drug_class
        self.weight_resistance_mechanism = weight_resistance_mechanism
        self.weight_amr_gene_family = weight_amr_gene_family

        # reduction='sum' + manual division by batch_size below, not the default
        # 'mean': BCEWithLogitsLoss's mean divides by batch_size * num_classes,
        # while CrossEntropyLoss's mean divides by batch_size only (its softmax +
        # NLL already collapses the class dimension per sample). With the default
        # reductions, drug_class_loss would end up ~num_classes times smaller than
        # the other two terms, so weight=1.0 each would not actually give equal
        # gradient contribution despite the locked-in alpha=beta=gamma=1 decision.
        # Summing then dividing by batch_size puts all three terms on the same
        # per-sample scale.
        self.drug_class_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        self.resistance_mechanism_loss_fn = nn.CrossEntropyLoss()
        self.amr_gene_family_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute per-task losses and their weighted sum.

        Args:
            logits: Output of ClassifierHead.forward() -- dict with keys
                'drug_class', 'resistance_mechanism', 'amr_gene_family'.
            batch: Batch dict as produced by AMRDataset's default collation --
                keys 'drug_class_labels', 'resistance_mechanism',
                'amr_gene_family'. The 'drug_class_labels' -> 'drug_class' name
                mismatch between AMRDataset and ClassifierHead is resolved here,
                rather than renaming either of those (both already ship with
                tests against their current key names).

        Returns:
            Dict with keys 'drug_class', 'resistance_mechanism', and
            'amr_gene_family' (each the unweighted per-task loss, for individual
            logging) plus 'total' (the weighted sum to call .backward() on).
        """
        batch_size = logits["drug_class"].shape[0]
        drug_class_loss = (
            self.drug_class_loss_fn(logits["drug_class"], batch["drug_class_labels"])
            / batch_size
        )
        resistance_mechanism_loss = self.resistance_mechanism_loss_fn(
            logits["resistance_mechanism"], batch["resistance_mechanism"]
        )
        amr_gene_family_loss = self.amr_gene_family_loss_fn(
            logits["amr_gene_family"], batch["amr_gene_family"]
        )

        total = (
            self.weight_drug_class * drug_class_loss
            + self.weight_resistance_mechanism * resistance_mechanism_loss
            + self.weight_amr_gene_family * amr_gene_family_loss
        )

        return {
            "drug_class": drug_class_loss,
            "resistance_mechanism": resistance_mechanism_loss,
            "amr_gene_family": amr_gene_family_loss,
            "total": total,
        }
