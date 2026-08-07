"""Thin CLI entry point for training (V1 or V2).

Equivalent to `python -m src.training.train --config <path>`; exists so the
CLAUDE.md-documented invocation below works directly.

Usage:
    python scripts/run_training.py --config configs/gpu_server_internal.yaml
    python scripts/run_training.py --config configs/gpu_task1_drugclass_internal.yaml
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable so `from src...` works when this script is
# run directly (`python scripts/run_training.py`), where Python would
# otherwise only put scripts/ itself on sys.path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train import train, train_v2
from src.utils.config import load_config


def main() -> None:
    """CLI entry point: python scripts/run_training.py --config <path>.

    Dispatches to train_v2() if the loaded config has a 'task' section (the
    four gpu_task{1,2}_*.yaml V2 configs), else falls back to train() (V1's
    fixed-architecture path) -- mirrors src.training.train.main()'s dispatch,
    which this script deliberately duplicates rather than calling (its own
    --config argument is parsed here, independently of that module's CLI).
    """
    parser = argparse.ArgumentParser(description="Train an AMR soft-prompting model (V1 or V2).")
    parser.add_argument(
        "--config", required=True, help="Path to a config YAML, e.g. configs/gpu_server_internal.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if "task" in config:
        train_v2(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
