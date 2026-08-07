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


def compute_single_target_metrics(
    target_name: str,
    batch_key: str,
    loss_type: str,
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute accuracy metric(s) for one configurable V2 target.

    Single-label targets ('ce', e.g. resistance_mechanism, amr_gene_family)
    get argmax accuracy, matching compute_metrics' existing behavior.
    Multi-label targets ('bce', e.g. drug_class) get 0.5-thresholded subset
    (exact-match) accuracy plus micro-F1, since plain accuracy isn't
    well-defined for multi-label predictions -- see
    src.data.dataset.TARGET_FIELD_SPECS for which fields use which loss_type.
    This is a training-time monitoring signal (per-batch, then epoch-averaged
    by run_v2_epoch); the holdout evaluation in evaluate.py computes true
    corpus-level macro/micro-F1 via sklearn instead of averaging this across
    batches.

    Args:
        target_name: Key logits are stored under (matches
            SingleTargetClassifierHead's output key).
        batch_key: Key the target's label lives under in an
            AMRDataset-collated batch (may differ from target_name, e.g.
            drug_class's multi-hot label is 'drug_class_labels').
        loss_type: 'bce' or 'ce'.
        logits: Output of SingleTargetClassifierHead.forward() -- dict with
            key target_name.
        batch: Batch dict as produced by AMRDataset's default collation --
            dict with key batch_key.

    Returns:
        'ce': dict with key '{target_name}_accuracy'.
        'bce': dict with keys '{target_name}_subset_accuracy' and
            '{target_name}_micro_f1'.

    Raises:
        ValueError: If loss_type isn't 'bce' or 'ce'.
    """
    target_logits = logits[target_name]
    target_labels = batch[batch_key]

    if loss_type == "ce":
        preds = target_logits.argmax(dim=-1)
        accuracy = (preds == target_labels).float().mean().item()
        return {f"{target_name}_accuracy": accuracy}

    if loss_type == "bce":
        preds = (torch.sigmoid(target_logits) > 0.5).float()
        subset_accuracy = (preds == target_labels).all(dim=-1).float().mean().item()

        true_positives = (preds * target_labels).sum()
        predicted_positive = preds.sum()
        actual_positive = target_labels.sum()
        precision = (true_positives / predicted_positive).item() if predicted_positive > 0 else 0.0
        recall = (true_positives / actual_positive).item() if actual_positive > 0 else 0.0
        micro_f1 = (
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )

        return {
            f"{target_name}_subset_accuracy": subset_accuracy,
            f"{target_name}_micro_f1": micro_f1,
        }

    raise ValueError(f"Unsupported loss_type {loss_type!r} -- must be 'bce' or 'ce'")
