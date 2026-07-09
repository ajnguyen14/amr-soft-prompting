"""One-time CARD preprocessing entry point.

Parses raw CARD files (FASTA + ARO index + card.json), builds label
vocabularies, and splits records into train/val/test — then serializes the
result to `output_dir` so downstream training never has to re-parse CARD or
re-derive the split. Per CLAUDE.md, this is the single script a fresh clone
runs to go from raw CARD files to all data artifacts needed for training; no
manual steps happen outside it.

Usage:
    python scripts/preprocess_card.py --config configs/cpu_server.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Make the project root importable so `from src...` works when this script is
# run directly (`python scripts/preprocess_card.py`), where Python would
# otherwise only put scripts/ itself on sys.path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.card_parser import get_label_vocabularies, load_card_dataset
from src.data.dataset import save_split_artifact, split_dataset
from src.utils.config import load_config

# Project-wide default (CLAUDE.md Reproducibility Requirements). Matches
# src/training/train.py's SEED so the split produced here is identical to the
# one train.py would derive on its own.
SEED = 42


def preprocess(config: dict[str, Any]) -> Path:
    """Parse, split, and serialize CARD data per `config`.

    Args:
        config: Merged config dict from load_config, with a 'paths' section
            (card_fasta, aro_index, card_json, output_dir).

    Returns:
        Path to the written artifact file.
    """
    records = load_card_dataset(
        config["paths"]["card_fasta"],
        config["paths"]["aro_index"],
        config["paths"].get("card_json"),
    )
    label_vocabularies = get_label_vocabularies(records)
    splits = split_dataset(records, seed=SEED)

    artifact_path = save_split_artifact(splits, label_vocabularies, config["paths"]["output_dir"])

    _print_summary(splits, label_vocabularies, artifact_path)
    return artifact_path


def _print_summary(
    splits: dict[str, list],
    label_vocabularies: dict[str, list[str]],
    artifact_path: Path,
) -> None:
    """Print record counts per split and vocab sizes to stdout.

    Args:
        splits: Dict with keys 'train', 'val', 'test' from split_dataset.
        label_vocabularies: Dict from get_label_vocabularies.
        artifact_path: Where the pickled artifact was written.
    """
    print(f"Wrote preprocessed CARD splits to: {artifact_path}")
    print("Record counts:")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {len(splits[split_name])}")
    print("Label vocabulary sizes:")
    for key, vocab in label_vocabularies.items():
        print(f"  {key}: {len(vocab)}")


def main() -> None:
    """CLI entry point: python scripts/preprocess_card.py --config <path>."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw CARD files into train/val/test split artifacts."
    )
    parser.add_argument(
        "--config", required=True, help="Path to a config YAML, e.g. configs/cpu_server.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess(config)


if __name__ == "__main__":
    main()
