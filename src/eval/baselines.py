"""Non-neural baselines for the V2 single-head tasks, scored on the same split as the models.

Usage:
    python -m src.eval.baselines --config configs/gpu_task3_genefamily_internal.yaml [--split test]

A trained model's accuracy is only meaningful relative to what trivial
predictors achieve on the identical split. Three baselines, CPU-only (no ESM-2):

* ``majority``: always predict the train-set majority label (the floor; shows
  how much class imbalance inflates accuracy).
* ``conditioning_lookup``: predict the train-set majority target for each
  conditioning value (e.g. gene_family -> most common drug_class). This is the
  label-leakage check from CLAUDE.md -- if it matches the model, the model is
  just decoding its soft-prompt input. Skipped when conditioning_field is
  'none' (the negative-control runs).
* ``nearest_neighbor``: copy the label of the most similar *train* sequence.
  Similarity is cosine over 3-mer counts -- a dependency-free proxy for BLAST
  identity, so treat its bins as approximate, not literal % identity.

The nearest-neighbor baseline also reports accuracy stratified by similarity to
the closest train sequence, which shows how much of any score comes from
near-duplicate homologs rather than generalization. Per-sequence similarities
are written out so a model's own predictions can be stratified the same way.

Splits are taken from the preprocessed artifact when present (same as
train.build_dataloaders), so baselines and models see identical train/test sets.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

from src.data.card_parser import CARDRecord, get_label_vocabularies, load_card_dataset
from src.data.dataset import TARGET_FIELD_SPECS, load_split_artifact, split_dataset
from src.training.train import SEED
from src.utils.config import load_config

KMER_SIZE = 3  # amino-acid k-mer length for the similarity proxy
# Upper edges of the similarity bins used to stratify nearest-neighbor accuracy.
SIMILARITY_BIN_EDGES = [0.5, 0.8, 0.95, 1.0001]

# CARDRecord attribute holding each single-label field's value.
RECORD_ATTR = {
    "amr_gene_family": "amr_gene_family",
    "resistance_mechanism": "resistance_mechanism",
    "ta_proximity": "ta_proximity_category",
}


def _majority(values: list[Any]) -> Any:
    """Return the most common value in `values` (ties broken by first seen)."""
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def _labelset(record: CARDRecord) -> frozenset[str]:
    """Return a record's drug classes as a hashable set."""
    return frozenset(record.drug_classes)


def _label_of(record: CARDRecord, target_field: str) -> Any:
    """Return the record's target label (str for single-label, frozenset for drug_class)."""
    if target_field == "drug_class":
        return _labelset(record)
    return getattr(record, RECORD_ATTR[target_field])


def _conditioning_of(record: CARDRecord, conditioning_field: str) -> str:
    """Return the record's conditioning value."""
    return getattr(record, RECORD_ATTR[conditioning_field])


def score_predictions(
    y_true: list[Any], y_pred: list[Any], target_field: str, vocab: list[str]
) -> dict[str, float]:
    """Score predictions with the same metric definitions evaluate.py uses.

    Single-label: accuracy + macro-F1 (over classes present in truth or
    prediction, matching evaluate_single_label_target). Multi-label: subset
    accuracy + micro/macro-F1 over the multi-hot drug_class vocabulary.

    Args:
        y_true: Per-sample true labels (str, or frozenset[str] for drug_class).
        y_pred: Per-sample predictions in the same form.
        target_field: Target being predicted.
        vocab: label_vocabularies[target_field].

    Returns:
        Dict of metric name to value.
    """
    if TARGET_FIELD_SPECS[target_field]["loss_type"] == "ce":
        return {
            "accuracy": float(np.mean([t == p for t, p in zip(y_true, y_pred)])),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def multi_hot(sets: list[frozenset[str]]) -> np.ndarray:
        index = {label: i for i, label in enumerate(vocab)}
        out = np.zeros((len(sets), len(vocab)), dtype=int)
        for row, labels in enumerate(sets):
            for label in labels:
                if label in index:
                    out[row, index[label]] = 1
        return out

    t, p = multi_hot(y_true), multi_hot(y_pred)
    return {
        "subset_accuracy": float((t == p).all(axis=1).mean()),
        "micro_f1": float(f1_score(t, p, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
    }


def nearest_neighbor_predictions(
    train: list[CARDRecord], test: list[CARDRecord], target_field: str
) -> tuple[list[Any], np.ndarray]:
    """Predict each test record's label from its most k-mer-similar train record.

    Args:
        train: Train-split records (the only labels the baseline may see).
        test: Records to predict.
        target_field: Target field whose label is copied from the neighbor.

    Returns:
        (predictions, max_similarity): the copied labels, and each test
        record's cosine similarity to its nearest train sequence.
    """
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(KMER_SIZE, KMER_SIZE), lowercase=False)
    x_train = normalize(vectorizer.fit_transform([r.sequence for r in train]))
    x_test = normalize(vectorizer.transform([r.sequence for r in test]))
    similarity = (x_test @ x_train.T).toarray()
    nearest = similarity.argmax(axis=1)
    return (
        [_label_of(train[i], target_field) for i in nearest],
        similarity.max(axis=1),
    )


def stratify_by_similarity(
    correct: np.ndarray, max_similarity: np.ndarray
) -> list[dict[str, float]]:
    """Accuracy of `correct` (bool per sample) within each nearest-train-similarity bin."""
    rows = []
    lower = 0.0
    for upper in SIMILARITY_BIN_EDGES:
        mask = (max_similarity >= lower) & (max_similarity < upper)
        rows.append(
            {
                "bin": f"[{lower:.2f}, {min(upper, 1.0):.2f}{']' if upper > 1 else ')'}",
                "n": int(mask.sum()),
                "accuracy": float(correct[mask].mean()) if mask.any() else float("nan"),
            }
        )
        lower = upper
    return rows


def run_baselines(
    splits: dict[str, list[CARDRecord]],
    label_vocabularies: dict[str, list[str]],
    conditioning_field: str,
    target_field: str,
    eval_split: str = "test",
) -> dict[str, Any]:
    """Compute all baselines for one task on `eval_split`, using only train labels.

    Args:
        splits: Dict with 'train'/'val'/'test' record lists.
        label_vocabularies: Vocabularies from get_label_vocabularies.
        conditioning_field: Soft-prompt conditioning field, or 'none'.
        target_field: Prediction target.
        eval_split: Split to score ('test' or 'val').

    Returns:
        Dict with per-baseline metrics, similarity-stratified nearest-neighbor
        accuracy (single-label targets only), and per-sample similarities.
    """
    train, evaluated = splits["train"], splits[eval_split]
    vocab = label_vocabularies[target_field]
    y_true = [_label_of(r, target_field) for r in evaluated]
    train_labels = [_label_of(r, target_field) for r in train]

    results: dict[str, Any] = {
        "task": {"conditioning_field": conditioning_field, "target_field": target_field},
        "split": eval_split,
        "n_train": len(train),
        "n_eval": len(evaluated),
    }

    majority = _majority(train_labels)
    results["majority"] = score_predictions(y_true, [majority] * len(y_true), target_field, vocab)

    if conditioning_field != "none":
        by_condition: dict[str, list[Any]] = {}
        for record, label in zip(train, train_labels):
            by_condition.setdefault(_conditioning_of(record, conditioning_field), []).append(label)
        lookup = {key: _majority(labels) for key, labels in by_condition.items()}
        predictions = [
            lookup.get(_conditioning_of(r, conditioning_field), majority) for r in evaluated
        ]
        results["conditioning_lookup"] = score_predictions(y_true, predictions, target_field, vocab)

    nn_predictions, max_similarity = nearest_neighbor_predictions(train, evaluated, target_field)
    results["nearest_neighbor"] = score_predictions(y_true, nn_predictions, target_field, vocab)
    exact = np.array([t == p for t, p in zip(y_true, nn_predictions)])
    results["nearest_neighbor_by_similarity"] = stratify_by_similarity(exact, max_similarity)
    results["max_train_similarity"] = {
        r.aro_accession: float(s) for r, s in zip(evaluated, max_similarity)
    }
    return results


def load_splits(config: dict[str, Any]) -> tuple[dict[str, list[CARDRecord]], dict[str, list[str]]]:
    """Load train/val/test splits the same way train.build_dataloaders does."""
    artifact = load_split_artifact(config["paths"]["output_dir"])
    if artifact is not None:
        return artifact
    records = load_card_dataset(
        config["paths"]["card_fasta"],
        config["paths"]["aro_index"],
        config["paths"].get("card_json"),
        ta_proximity_path=config["paths"].get("ta_proximity_results"),
    )
    return split_dataset(records, seed=SEED), get_label_vocabularies(records)


def main() -> None:
    """CLI entry point: python -m src.eval.baselines --config <task config> [--split test]."""
    parser = argparse.ArgumentParser(description="Score non-neural baselines for a V2 task config.")
    parser.add_argument("--config", required=True, help="Task config, e.g. configs/gpu_task1_*.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    splits, label_vocabularies = load_splits(config)
    results = run_baselines(
        splits,
        label_vocabularies,
        config["task"]["conditioning_field"],
        config["task"]["target_field"],
        args.split,
    )

    out_dir = Path(config["paths"]["output_dir"]) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baselines_{results['task']['target_field']}_"
    out_path = out_path.with_name(
        out_path.name + f"{results['task']['conditioning_field']}_{args.split}.json"
    )
    out_path.write_text(json.dumps(results, indent=2))

    for name in ("majority", "conditioning_lookup", "nearest_neighbor"):
        if name in results:
            print(f"{name}: {results[name]}")
    print("nearest_neighbor by similarity to closest train sequence:")
    for row in results["nearest_neighbor_by_similarity"]:
        print(f"  {row['bin']}: n={row['n']}, accuracy={row['accuracy']:.4f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
