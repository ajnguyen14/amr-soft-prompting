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

from src.eval.metrics import compute_metrics
from src.models.classifier import ClassifierHead
from src.models.esm2_wrapper import ESM2Wrapper
from src.models.soft_prompt import SoftPromptModule
from src.training.train import SEED, build_dataloaders, build_models, move_batch_to_device, set_seed
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


def main() -> None:
    """CLI entry point: python -m src.eval.evaluate --config <path> [--checkpoint <path>]."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained V1 AMR soft-prompting checkpoint on the CARD test holdout."
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

    results = evaluate(config, checkpoint_path)

    print(f"Evaluation complete for checkpoint {checkpoint_path} (epoch {results['checkpoint_epoch']}).")
    print(f"Artifacts written to {results['eval_dir']}")
    print(f"Aggregate metrics: {results['aggregate']}")
    print(
        "amr_gene_family: accuracy="
        f"{results['amr_gene_family']['accuracy']:.4f}, macro_f1={results['amr_gene_family']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
