"""Holdout evaluation for a trained V1 AMR soft-prompting checkpoint.

Usage:
    python -m src.eval.evaluate --config configs/gpu_server_internal.yaml \\
        --checkpoint outputs/best_model.pt

Runs the frozen ESM-2 + soft prompt + classifier pipeline over the CARD test
split (never train/val) and writes a self-contained JSON results file plus a
raw confusion-matrix CSV for the amr_gene_family task to `<output_dir>/eval/`.
resistance_mechanism and drug_class are not evaluated here -- both are fed
into the soft prompt as conditioning input rather than predicted, so scoring
them would just measure how well the classifier decodes its own soft-prompt
embedding (see docs/STATUS.md's label-leakage note). Builds on
`src.eval.metrics.compute_metrics` for aggregate numbers rather than
recomputing accuracy/F1 a second way, and reuses `src.training.train`'s
dataloader/model construction so eval and training never diverge on how data
or models are built.

No wandb logging here -- evaluation is a point-in-time report against an
already-logged training run, not itself a run to log.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.data.dataset import TARGET_FIELD_SPECS
from src.eval.metrics import compute_metrics, compute_single_target_metrics
from src.models.classifier import ClassifierHead, SingleTargetClassifierHead
from src.models.esm2_wrapper import ESM2Wrapper
from src.models.soft_prompt import SingleFieldSoftPrompt, SoftPromptModule
from src.training.train import (
    SEED,
    build_dataloaders,
    build_models,
    build_v2_models,
    move_batch_to_device,
    set_seed,
)
from src.utils.config import load_config

# amr_gene_family has 398 classes -- too many for a readable confusion
# matrix, so it gets aggregate accuracy/macro-F1 plus this many top-confused
# (true, predicted) pairs instead of a rendered matrix.
TOP_CONFUSED_PAIRS = 10


@torch.no_grad()
def collect_predictions(
    loader: DataLoader,
    esm2: ESM2Wrapper,
    soft_prompt: SoftPromptModule,
    classifier: ClassifierHead,
    device: str,
) -> dict[str, torch.Tensor]:
    """Run the frozen pipeline over `loader` and concatenate every prediction/label.

    Unlike train.py's run_epoch (which only needs epoch-averaged scalars),
    confusion matrices need every individual prediction, so this concatenates
    raw logits and labels across the whole loader instead of aggregating
    per-batch metrics.

    Args:
        loader: DataLoader yielding AMRDataset-collated batches (the test split).
        esm2, soft_prompt, classifier: Model components, already moved to `device`
            with trained weights loaded.
        device: Torch device string.

    Returns:
        Dict with keys 'amr_gene_family_logits' and 'amr_gene_family_labels',
        each a single CPU tensor concatenated over the full loader.
    """
    esm2.eval()
    soft_prompt.eval()
    classifier.eval()

    collected: dict[str, list[torch.Tensor]] = {
        "amr_gene_family_logits": [],
        "amr_gene_family_labels": [],
    }

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        soft_prompt_vectors = soft_prompt(batch["resistance_mechanism"], batch["drug_class_labels"])
        pooled = esm2(batch["sequence"], soft_prompt_vectors)
        logits = classifier(pooled)

        collected["amr_gene_family_logits"].append(logits["amr_gene_family"].cpu())
        collected["amr_gene_family_labels"].append(batch["amr_gene_family"].cpu())

    return {key: torch.cat(tensors, dim=0) for key, tensors in collected.items()}


def top_confused_pairs(cm: np.ndarray, labels: list[str], top_n: int) -> list[dict[str, Any]]:
    """Find the `top_n` off-diagonal (true, predicted) pairs with the most misclassifications.

    Args:
        cm: (num_labels, num_labels) confusion matrix, rows=true, cols=predicted.
        labels: Class names in the same order as `cm`'s rows/columns.
        top_n: Number of pairs to return.

    Returns:
        List of dicts with keys 'true', 'predicted', 'count', sorted descending
        by count. Diagonal (correct) entries are excluded.
    """
    pairs = [
        (int(cm[i, j]), labels[i], labels[j])
        for i in range(cm.shape[0])
        for j in range(cm.shape[1])
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda entry: entry[0], reverse=True)
    return [
        {"true": true_label, "predicted": pred_label, "count": count}
        for count, true_label, pred_label in pairs[:top_n]
    ]


def evaluate_amr_gene_family(
    predictions: dict[str, torch.Tensor], vocab: list[str], eval_dir: Path, accuracy: float
) -> dict[str, Any]:
    """Macro-F1 + top-confused pairs for the 398-class amr_gene_family task.

    The full confusion matrix is dumped as CSV for offline analysis but never
    rendered -- a 398x398 heatmap isn't readable or poster-usable.

    Args:
        predictions: Output of collect_predictions.
        vocab: label_vocabularies['amr_gene_family'] (398 classes).
        eval_dir: Directory to write the raw confusion-matrix CSV into.
        accuracy: amr_gene_family accuracy, as already computed by
            compute_metrics in evaluate() -- passed in rather than
            recomputed here so the two never diverge.

    Returns:
        Dict with keys 'accuracy', 'macro_f1', 'top_confused_pairs', and
        'confusion_matrix_csv' (path to the dumped CSV).
    """
    y_true = predictions["amr_gene_family_labels"].numpy()
    y_pred = predictions["amr_gene_family_logits"].argmax(dim=-1).numpy()

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(vocab))))

    csv_path = eval_dir / "confusion_matrix_amr_gene_family.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([""] + vocab)
        for label, row in zip(vocab, cm):
            writer.writerow([label] + row.tolist())

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "top_confused_pairs": top_confused_pairs(cm, vocab, TOP_CONFUSED_PAIRS),
        "confusion_matrix_csv": str(csv_path),
    }


def evaluate(config: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    """Run the full V1 holdout evaluation and write all artifacts to disk.

    Args:
        config: Merged config dict from load_config (base.yaml + an
            environment override), used to rebuild the exact same test split,
            label vocabularies, and model architecture the checkpoint was
            trained with.
        checkpoint_path: Path to a .pt file saved by src.training.train
            (keys 'soft_prompt_state_dict', 'classifier_state_dict', 'epoch').

    Returns:
        The same results dict written to '<output_dir>/eval/.../evaluation_results.json':
        keys 'checkpoint', 'checkpoint_epoch', 'eval_dir', 'config', 'timestamp',
        'aggregate', 'amr_gene_family'.
    """
    set_seed(SEED)
    device = config["model"]["device"]

    _train_loader, _val_loader, test_loader, label_vocabularies = build_dataloaders(config)
    esm2, soft_prompt, classifier, _loss_fn = build_models(config, label_vocabularies, device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    soft_prompt.load_state_dict(checkpoint["soft_prompt_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])

    predictions = collect_predictions(test_loader, esm2, soft_prompt, classifier, device)

    aggregate = compute_metrics(
        logits={"amr_gene_family": predictions["amr_gene_family_logits"]},
        batch={"amr_gene_family": predictions["amr_gene_family_labels"]},
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    eval_dir = (
        Path(config["paths"]["output_dir"])
        / "eval"
        / f"{checkpoint_path.stem}_{timestamp.replace(':', '-')}"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "eval_dir": str(eval_dir),
        "config": config,
        "timestamp": timestamp,
        "aggregate": aggregate,
        "amr_gene_family": evaluate_amr_gene_family(
            predictions,
            label_vocabularies["amr_gene_family"],
            eval_dir,
            aggregate["amr_gene_family_accuracy"],
        ),
    }

    with open(eval_dir / "evaluation_results.json", "w") as results_file:
        json.dump(results, results_file, indent=2)

    return results


@torch.no_grad()
def collect_predictions_v2(
    loader: DataLoader,
    esm2: ESM2Wrapper,
    soft_prompt: SingleFieldSoftPrompt,
    classifier: SingleTargetClassifierHead,
    conditioning_field: str,
    device: str,
) -> dict[str, torch.Tensor]:
    """Run the frozen V2 pipeline over `loader` and concatenate every prediction/label.

    Generalizes collect_predictions (V1, fixed to the amr_gene_family key) to
    whichever single target classifier.target_name is configured for, per
    CLAUDE.md's Single-Head Architecture table.

    Args:
        loader: DataLoader yielding AMRDataset-collated batches (the test split).
        esm2, soft_prompt, classifier: Model components from build_v2_models,
            already moved to `device` with trained weights loaded.
        conditioning_field: Vocab key feeding the soft prompt; its AMRDataset
            batch key is looked up via TARGET_FIELD_SPECS.
        device: Torch device string.

    Returns:
        Dict with keys '<target_name>_logits' and '<target_name>_labels',
        each a single CPU tensor concatenated over the full loader.
    """
    esm2.eval()
    soft_prompt.eval()
    classifier.eval()

    conditioning_batch_key = TARGET_FIELD_SPECS[conditioning_field]["batch_key"]
    target_name = classifier.target_name
    target_batch_key = TARGET_FIELD_SPECS[target_name]["batch_key"]

    logits_batches: list[torch.Tensor] = []
    labels_batches: list[torch.Tensor] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        soft_prompt_vectors = soft_prompt(batch[conditioning_batch_key])
        pooled = esm2(batch["sequence"], soft_prompt_vectors)
        logits = classifier(pooled)

        logits_batches.append(logits[target_name].cpu())
        labels_batches.append(batch[target_batch_key].cpu())

    return {
        f"{target_name}_logits": torch.cat(logits_batches, dim=0),
        f"{target_name}_labels": torch.cat(labels_batches, dim=0),
    }


def evaluate_single_label_target(
    predictions: dict[str, torch.Tensor],
    target_name: str,
    vocab: list[str],
    eval_dir: Path,
    accuracy: float,
) -> dict[str, Any]:
    """Macro-F1 + top-confused pairs for a single-label ('ce') V2 target.

    Generalizes evaluate_amr_gene_family (V1) to any single-label V2 target
    (resistance_mechanism for Run 2, amr_gene_family for Run 3).

    Args:
        predictions: Output of collect_predictions_v2.
        target_name: e.g. 'resistance_mechanism'.
        vocab: label_vocabularies[target_name].
        eval_dir: Directory to write the raw confusion-matrix CSV into.
        accuracy: Already-computed accuracy (from compute_single_target_metrics
            in evaluate_v2), passed in rather than recomputed here so the two
            never diverge.

    Returns:
        Dict with keys 'accuracy', 'macro_f1', 'top_confused_pairs', and
        'confusion_matrix_csv' (path to the dumped CSV).
    """
    y_true = predictions[f"{target_name}_labels"].numpy()
    y_pred = predictions[f"{target_name}_logits"].argmax(dim=-1).numpy()

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(vocab))))

    csv_path = eval_dir / f"confusion_matrix_{target_name}.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([""] + vocab)
        for label, row in zip(vocab, cm):
            writer.writerow([label] + row.tolist())

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "top_confused_pairs": top_confused_pairs(cm, vocab, TOP_CONFUSED_PAIRS),
        "confusion_matrix_csv": str(csv_path),
    }


def evaluate_multi_label_target(
    predictions: dict[str, torch.Tensor],
    target_name: str,
    vocab: list[str],
    eval_dir: Path,
) -> dict[str, Any]:
    """Subset accuracy + micro/macro-F1 + per-class F1 for a multi-label ('bce') V2 target.

    Plain argmax accuracy / a full confusion matrix don't apply to
    multi-label predictions (Run 1's drug_class, BCEWithLogitsLoss) -- see
    CLAUDE.md's Single-Head Architecture table. Predictions are
    0.5-thresholded sigmoid outputs, matching training-time
    SingleTargetLoss/compute_single_target_metrics so eval-time and
    train-time predictions are defined identically.

    Args:
        predictions: Output of collect_predictions_v2.
        target_name: e.g. 'drug_class'.
        vocab: label_vocabularies[target_name].
        eval_dir: Directory to write the per-class F1 CSV into.

    Returns:
        Dict with keys 'subset_accuracy', 'micro_f1', 'macro_f1', and
        'per_class_f1_csv' (path to the dumped CSV).
    """
    y_true = predictions[f"{target_name}_labels"].numpy()
    y_score = torch.sigmoid(predictions[f"{target_name}_logits"]).numpy()
    y_pred = (y_score > 0.5).astype(y_true.dtype)

    subset_accuracy = float((y_pred == y_true).all(axis=-1).mean())
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    csv_path = eval_dir / f"per_class_f1_{target_name}.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class", "f1"])
        for label, score in zip(vocab, per_class_f1):
            writer.writerow([label, float(score)])

    return {
        "subset_accuracy": subset_accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_class_f1_csv": str(csv_path),
    }


def evaluate_v2(config: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    """Run a full V2 holdout evaluation and write all artifacts to disk.

    Generalizes evaluate() (V1) to any of Runs 1-3, per config['task'];
    dispatches to evaluate_single_label_target or evaluate_multi_label_target
    based on the target field's loss_type (TARGET_FIELD_SPECS).

    Args:
        config: Merged config dict from load_config, with a 'task' section,
            used to rebuild the exact same test split, label vocabularies,
            and model architecture the checkpoint was trained with.
        checkpoint_path: Path to a .pt file saved by
            src.training.train.train_v2 (keys 'soft_prompt_state_dict',
            'classifier_state_dict', 'epoch').

    Returns:
        Dict written to '<output_dir>/eval/.../evaluation_results.json':
        keys 'checkpoint', 'checkpoint_epoch', 'eval_dir', 'config',
        'timestamp', 'aggregate', and the target field name (mapping to
        evaluate_single_label_target's or evaluate_multi_label_target's dict).
    """
    set_seed(SEED)
    device = config["model"]["device"]
    conditioning_field = config["task"]["conditioning_field"]
    target_field = config["task"]["target_field"]
    loss_type = TARGET_FIELD_SPECS[target_field]["loss_type"]

    _train_loader, _val_loader, test_loader, label_vocabularies = build_dataloaders(config)
    esm2, soft_prompt, classifier, _loss_fn = build_v2_models(config, label_vocabularies, device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    soft_prompt.load_state_dict(checkpoint["soft_prompt_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])

    predictions = collect_predictions_v2(
        test_loader, esm2, soft_prompt, classifier, conditioning_field, device
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    eval_dir = (
        Path(config["paths"]["output_dir"])
        / "eval"
        / f"{checkpoint_path.stem}_{timestamp.replace(':', '-')}"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    if loss_type == "ce":
        target_batch_key = TARGET_FIELD_SPECS[target_field]["batch_key"]
        aggregate = compute_single_target_metrics(
            target_field,
            target_batch_key,
            loss_type,
            logits={target_field: predictions[f"{target_field}_logits"]},
            batch={target_batch_key: predictions[f"{target_field}_labels"]},
        )
        target_results = evaluate_single_label_target(
            predictions,
            target_field,
            label_vocabularies[target_field],
            eval_dir,
            aggregate[f"{target_field}_accuracy"],
        )
    else:
        target_results = evaluate_multi_label_target(
            predictions, target_field, label_vocabularies[target_field], eval_dir
        )
        aggregate = {
            f"{target_field}_subset_accuracy": target_results["subset_accuracy"],
            f"{target_field}_micro_f1": target_results["micro_f1"],
        }

    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "eval_dir": str(eval_dir),
        "config": config,
        "timestamp": timestamp,
        "aggregate": aggregate,
        target_field: target_results,
    }

    with open(eval_dir / "evaluation_results.json", "w") as results_file:
        json.dump(results, results_file, indent=2)

    return results


def main() -> None:
    """CLI entry point: python -m src.eval.evaluate --config <path> [--checkpoint <path>].

    Dispatches to evaluate_v2() if the loaded config has a 'task' section
    (the four gpu_task{1,2}_*.yaml configs), else falls back to evaluate()
    (V1's fixed-architecture path) -- mirrors train.py's main() dispatch, so
    the same --config file that trained a checkpoint also evaluates it
    correctly with no extra flag.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained AMR soft-prompting checkpoint (V1 or V2) on the CARD test holdout."
    )
    parser.add_argument(
        "--config", required=True, help="Path to a config YAML, e.g. configs/gpu_server_internal.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint .pt file. Defaults to '<output_dir>/best_model.pt' from --config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config["paths"]["output_dir"]) / "best_model.pt"
    )

    if "task" in config:
        results = evaluate_v2(config, checkpoint_path)
        target_field = config["task"]["target_field"]
        print(
            f"Evaluation complete for checkpoint {checkpoint_path} "
            f"(epoch {results['checkpoint_epoch']})."
        )
        print(f"Artifacts written to {results['eval_dir']}")
        print(f"Target field: {target_field}")
        print(f"Aggregate metrics: {results['aggregate']}")
    else:
        results = evaluate(config, checkpoint_path)
        print(
            f"Evaluation complete for checkpoint {checkpoint_path} "
            f"(epoch {results['checkpoint_epoch']})."
        )
        print(f"Artifacts written to {results['eval_dir']}")
        print(f"Aggregate metrics: {results['aggregate']}")
        print(
            "amr_gene_family: accuracy="
            f"{results['amr_gene_family']['accuracy']:.4f}, "
            f"macro_f1={results['amr_gene_family']['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
