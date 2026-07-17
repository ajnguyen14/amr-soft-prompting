"""Shared per-task metric computation, used by both train.py's per-epoch
monitoring and evaluate.py's holdout evaluation.
"""

import torch


def compute_metrics(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute accuracy for the amr_gene_family task.

    resistance_mechanism and drug_class are not scored here -- both are fed
    into SoftPromptModule as conditioning input, so scoring predictions of
    them would only measure how well the classifier decodes its own
    soft-prompt embedding (see docs/STATUS.md's label-leakage note).

    Args:
        logits: Output of ClassifierHead.forward() -- dict with key
            'amr_gene_family'.
        batch: Batch dict as produced by AMRDataset's default collation --
            key 'amr_gene_family'.

    Returns:
        Dict with key:
            amr_gene_family_accuracy: fraction of argmax predictions matching
                the label.
    """
    amr_gene_family_pred = logits["amr_gene_family"].argmax(dim=-1)
    amr_gene_family_acc = (
        (amr_gene_family_pred == batch["amr_gene_family"]).float().mean().item()
    )

    return {
        "amr_gene_family_accuracy": amr_gene_family_acc,
    }
