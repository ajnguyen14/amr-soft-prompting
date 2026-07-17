"""Main training loop for AMR soft prompting (V1).

Usage:
    python -m src.training.train --config configs/gpu_server_internal.yaml
"""

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from src.data.card_parser import CARDRecord, get_label_vocabularies, load_card_dataset
from src.data.dataset import AMRDataset, load_split_artifact, split_dataset
from src.eval.metrics import compute_metrics
from src.models.classifier import ClassifierHead
from src.models.esm2_wrapper import ESM2Wrapper
from src.models.soft_prompt import SoftPromptModule
from src.training.loss import AMRLoss
from src.utils.config import load_config

# Project-wide default (CLAUDE.md Reproducibility Requirements).
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python's random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dataloaders_from_splits(
    splits: dict[str, list[CARDRecord]],
    label_vocabularies: dict[str, list[str]],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, list[str]]]:
    """Build train/val/test DataLoaders from an already-computed split.

    Args:
        splits: Dict with keys 'train', 'val', 'test', e.g. from split_dataset
            or a preprocessed split artifact.
        label_vocabularies: Dict from get_label_vocabularies.
        batch_size: Batch size shared by all three DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, label_vocabularies). Only
        train_loader shuffles.
    """
    loaders = {}
    for split_name in ("train", "val", "test"):
        dataset = AMRDataset(splits[split_name], label_vocabularies)
        loaders[split_name] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split_name == "train")
        )

    return loaders["train"], loaders["val"], loaders["test"], label_vocabularies


def build_dataloaders_from_records(
    records: list[CARDRecord],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, list[str]]]:
    """Build train/val/test DataLoaders from already-loaded CARDRecord objects.

    Label vocabularies are built from the full record set before splitting so
    val/test never contain a label absent from train's vocabulary.

    Args:
        records: List of CARDRecord, e.g. from load_card_dataset.
        batch_size: Batch size shared by all three DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, label_vocabularies). Only
        train_loader shuffles.
    """
    label_vocabularies = get_label_vocabularies(records)
    splits = split_dataset(records, seed=SEED)
    return _dataloaders_from_splits(splits, label_vocabularies, batch_size)


def build_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, list[str]]]:
    """Build DataLoaders for `config`, preferring a preprocessed split artifact.

    If scripts/preprocess_card.py has already been run for this config's
    output_dir, its pickled split + label vocabularies are loaded directly so
    training never re-parses raw CARD or re-derives the split. Otherwise falls
    back to parsing config's paths directly (e.g. for quick local iteration
    without running preprocessing first).

    Args:
        config: Merged config dict from load_config, with a 'paths' section
            (card_fasta, aro_index, card_json, output_dir) and
            'training.batch_size'.

    Returns:
        Same as build_dataloaders_from_records.
    """
    artifact = load_split_artifact(config["paths"]["output_dir"])
    if artifact is not None:
        splits, label_vocabularies = artifact
        return _dataloaders_from_splits(splits, label_vocabularies, config["training"]["batch_size"])

    records = load_card_dataset(
        config["paths"]["card_fasta"],
        config["paths"]["aro_index"],
        config["paths"].get("card_json"),
    )
    return build_dataloaders_from_records(records, config["training"]["batch_size"])


def build_models(
    config: dict[str, Any],
    label_vocabularies: dict[str, list[str]],
    device: str,
) -> tuple[ESM2Wrapper, SoftPromptModule, ClassifierHead, AMRLoss]:
    """Construct the frozen ESM-2 backbone, soft prompt, classifier, and loss.

    Args:
        config: Merged config dict with 'model', 'classifier', and 'loss' sections.
        label_vocabularies: Dict from get_label_vocabularies, used to size the
            soft prompt embeddings and classifier heads.
        device: Torch device string (e.g. 'cuda' or 'cpu') to move all
            trainable modules to. ESM2Wrapper's own device follows from its
            frozen weights, which are moved along with it.

    Returns:
        (esm2, soft_prompt, classifier, loss_fn), all moved to `device`.
    """
    num_mechanisms = len(label_vocabularies["resistance_mechanism"])
    num_drug_classes = len(label_vocabularies["drug_class"])
    num_families = len(label_vocabularies["amr_gene_family"])

    injection_mode = config["model"]["injection_mode"]
    esm2 = ESM2Wrapper(config["model"]["esm2_variant"], injection_mode=injection_mode).to(device)

    soft_prompt = SoftPromptModule(num_mechanisms, num_drug_classes, esm2.embed_dim).to(device)

    classifier = ClassifierHead(
        input_dim=esm2.output_dim(SoftPromptModule.NUM_PROMPT_TOKENS),
        hidden_dim=config["classifier"]["hidden_dim"],
        dropout=config["classifier"]["dropout"],
        num_families=num_families,
    ).to(device)

    loss_fn = AMRLoss(**config["loss"]).to(device)

    return esm2, soft_prompt, classifier, loss_fn


def build_optimizer(
    name: str,
    params: list[torch.nn.Parameter],
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Map a config optimizer name string to a torch.optim.Optimizer instance.

    Args:
        name: Optimizer name from config['training']['optimizer'] (case-insensitive).
        params: Trainable parameters to optimize (soft prompt + classifier only —
            ESM-2 is always frozen and never included here).
        learning_rate: Learning rate from config['training']['learning_rate'].

    Returns:
        A constructed torch.optim.Optimizer.

    Raises:
        ValueError: If `name` isn't a supported V1 optimizer.
    """
    if name.lower() == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    raise ValueError(f"Unsupported optimizer {name!r} — only 'adam' is implemented for V1")


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    """Move tensor entries of a collated batch dict to device; leave others as-is.

    'sequence' stays a list[str] (ESM2Wrapper tokenizes internally), and
    'aro_accession' stays a list[str] — neither is a tensor.
    """
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(
    loader: DataLoader,
    esm2: ESM2Wrapper,
    soft_prompt: SoftPromptModule,
    classifier: ClassifierHead,
    loss_fn: AMRLoss,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Run one epoch over `loader`. Trains if `optimizer` is given, else only evaluates.

    ESM-2 is always kept in eval() mode regardless of train/val — it's frozen,
    so there's no reason to let its internal dropout inject noise into a
    representation nothing is learning from; only soft_prompt and classifier
    switch train()/eval() mode.

    Args:
        loader: DataLoader yielding AMRDataset-collated batches.
        esm2, soft_prompt, classifier, loss_fn: The model components, already
            moved to `device`.
        device: Torch device string.
        optimizer: If given, this is a training epoch: backward + step per
            batch. If None, this is an evaluation epoch: no gradient updates.

    Returns:
        Dict of epoch-averaged values: 'total', 'amr_gene_family' (loss), plus
        'amr_gene_family_accuracy' from compute_metrics.
    """
    is_train = optimizer is not None
    soft_prompt.train(is_train)
    classifier.train(is_train)
    esm2.eval()

    totals = {
        "total": 0.0,
        "amr_gene_family": 0.0,
        "amr_gene_family_accuracy": 0.0,
    }
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = move_batch_to_device(batch, device)

            soft_prompt_vectors = soft_prompt(
                batch["resistance_mechanism"], batch["drug_class_labels"]
            )
            pooled = esm2(batch["sequence"], soft_prompt_vectors)
            logits = classifier(pooled)
            losses = loss_fn(logits, batch)

            if is_train:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

            metrics = compute_metrics(logits, batch)

            totals["total"] += losses["total"].item()
            totals["amr_gene_family"] += losses["amr_gene_family"].item()
            totals["amr_gene_family_accuracy"] += metrics["amr_gene_family_accuracy"]
            n_batches += 1

    return {key: value / n_batches for key, value in totals.items()}


def train(config: dict[str, Any]) -> None:
    """Run the full V1 training loop.

    Trains and validates every epoch, logs the amr_gene_family loss plus total
    loss and accuracy to wandb, and checkpoints only on a new best total
    validation loss. ESM-2's own weights are never checkpointed — they're
    frozen and fully determined by config['model']['esm2_variant'].

    Args:
        config: Merged config dict from load_config (base.yaml + an
            environment override, e.g. configs/gpu_server_internal.yaml).
    """
    set_seed(SEED)

    device = config["model"]["device"]
    train_loader, val_loader, _test_loader, label_vocabularies = build_dataloaders(config)
    esm2, soft_prompt, classifier, loss_fn = build_models(config, label_vocabularies, device)

    trainable_params = list(soft_prompt.parameters()) + list(classifier.parameters())
    optimizer = build_optimizer(
        config["training"]["optimizer"], trainable_params, config["training"]["learning_rate"]
    )

    # gradient_checkpointing is derived from injection_mode (see esm2_wrapper.py),
    # not an independent hyperparameter — logged for run-to-run visibility only,
    # so it's added to the wandb snapshot without mutating the loaded config.
    wandb_config = {
        **config,
        "model": {
            **config["model"],
            "gradient_checkpointing": config["model"]["injection_mode"] == "internal",
        },
    }
    wandb.init(
        project=config["logging"]["wandb_project"],
        name=config["logging"]["wandb_run_name"],
        config=wandb_config,
    )

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"

    best_val_loss = float("inf")
    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        train_metrics = run_epoch(
            train_loader, esm2, soft_prompt, classifier, loss_fn, device, optimizer
        )
        val_metrics = run_epoch(
            val_loader, esm2, soft_prompt, classifier, loss_fn, device, optimizer=None
        )

        wandb.log(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
            }
        )

        # Checkpoint on best total val loss only (V1 default). Revisit this
        # criterion in V2 if the per-term loss logging above shows one task
        # dominating the total.
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "soft_prompt_state_dict": soft_prompt.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

    wandb.finish()


def main() -> None:
    """CLI entry point: python -m src.training.train --config <path>."""
    parser = argparse.ArgumentParser(description="Train the V1 AMR soft-prompting model.")
    parser.add_argument(
        "--config", required=True, help="Path to a config YAML, e.g. configs/gpu_server_internal.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
