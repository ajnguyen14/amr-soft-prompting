"""Thin CLI entry point for V1 training.

Equivalent to `python -m src.training.train --config <path>`; exists so the
CLAUDE.md-documented invocation below works directly.

Usage:
    python scripts/run_training.py --config configs/gpu_server_internal.yaml
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable so `from src...` works when this script is
# run directly (`python scripts/run_training.py`), where Python would
# otherwise only put scripts/ itself on sys.path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train import train
from src.utils.config import load_config


def main() -> None:
    """CLI entry point: python scripts/run_training.py --config <path>."""
    parser = argparse.ArgumentParser(description="Train the V1 AMR soft-prompting model.")
    parser.add_argument(
        "--config", required=True, help="Path to a config YAML, e.g. configs/gpu_server_internal.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
