"""Combined multi-task loss for AMR gene classification."""

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
