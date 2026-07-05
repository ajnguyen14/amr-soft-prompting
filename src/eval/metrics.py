"""Shared per-task metric computation, used by both train.py's per-epoch
monitoring and evaluate.py's holdout evaluation.
"""

import torch
from sklearn.metrics import f1_score

# Sigmoid probability above which a drug_class logit counts as a positive
# prediction. Named rather than inline so both train.py and evaluate.py apply
# the same threshold.
DRUG_CLASS_THRESHOLD = 0.5


def compute_metrics(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute accuracy/F1 metrics for the three AMR classification tasks.

    Args:
        logits: Output of ClassifierHead.forward() -- dict with keys
            'drug_class', 'resistance_mechanism', 'amr_gene_family'.
        batch: Batch dict as produced by AMRDataset's default collation --
            keys 'drug_class_labels', 'resistance_mechanism', 'amr_gene_family'.

    Returns:
        Dict with keys:
            resistance_mechanism_accuracy, amr_gene_family_accuracy: fraction
                of argmax predictions matching the label.
            drug_class_f1_micro: micro-averaged F1 over the multi-hot
                drug_class vocabulary, thresholded at DRUG_CLASS_THRESHOLD.
    """
    resistance_mechanism_pred = logits["resistance_mechanism"].argmax(dim=-1)
    resistance_mechanism_acc = (
        (resistance_mechanism_pred == batch["resistance_mechanism"]).float().mean().item()
    )

    amr_gene_family_pred = logits["amr_gene_family"].argmax(dim=-1)
    amr_gene_family_acc = (
        (amr_gene_family_pred == batch["amr_gene_family"]).float().mean().item()
    )

    drug_class_probs = torch.sigmoid(logits["drug_class"])
    drug_class_pred = (drug_class_probs > DRUG_CLASS_THRESHOLD).float()
    drug_class_f1 = f1_score(
        batch["drug_class_labels"].detach().cpu().numpy(),
        drug_class_pred.detach().cpu().numpy(),
        average="micro",
        zero_division=0,
    )

    return {
        "resistance_mechanism_accuracy": resistance_mechanism_acc,
        "amr_gene_family_accuracy": amr_gene_family_acc,
        "drug_class_f1_micro": float(drug_class_f1),
    }
