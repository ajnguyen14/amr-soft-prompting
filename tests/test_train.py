"""Smoke tests for the training loop (src/training/train.py).

Uses the 8M model on CPU with a small synthetic CARD-like dataset (tiny FASTA +
ARO index written to tmp_path), per CLAUDE.md's smoke-test convention. Never
touches the real CARD dataset or the 650M model -- that's a GPU-server-only
training run, out of scope for a CPU smoke test.
"""

import textwrap
from pathlib import Path
from typing import Any

import pytest
import torch

from src.training.train import (
    build_dataloaders_from_records,
    build_models,
    build_optimizer,
    run_epoch,
    train,
)
from src.data.card_parser import load_card_dataset

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
NUM_ACCESSIONS_PER_MECHANISM = 10  # -> 8/1/1 train/val/test split per mechanism


def _make_card_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write a small synthetic FASTA + ARO index: 2 mechanisms x 10 accessions each.

    Enough accessions per mechanism for split_dataset's default 80/10/10 to
    produce a non-empty train, val, and test split (see test_dataset.py's
    stratified_records fixture for the same reasoning).
    """
    mechanisms = ["antibiotic inactivation", "antibiotic target alteration"]
    fasta_lines = []
    tsv_lines = [
        "ARO Accession\tCVTERM ID\tModel Sequence ID\tModel ID\tModel Name\tARO Name\t"
        "Protein Accession\tDNA Accession\tAMR Gene Family\tDrug Class\t"
        "Resistance Mechanism\tCARD Short Name"
    ]

    aro_id = 3000000
    for mech_idx, mechanism in enumerate(mechanisms):
        for i in range(NUM_ACCESSIONS_PER_MECHANISM):
            aro_id += 1
            protein_acc = f"PROT{mech_idx}{i:03d}.1"
            gene_name = f"gene{mech_idx}_{i}"
            fasta_lines.append(f">gb|{protein_acc}|ARO:{aro_id}|{gene_name} [test organism]")
            fasta_lines.append("MKAYFIAILTLFTCIATVVRAQQMSELENRIDSLLNGK")
            tsv_lines.append(
                f"ARO:{aro_id}\t{aro_id}\t1\t1\t{gene_name}\t{gene_name}\t{protein_acc}\t"
                f"DNA{aro_id}\tfamily_{mech_idx}\tsome antibiotic\t{mechanism}\t{gene_name}"
            )

    fasta_path = tmp_path / "test.fasta"
    tsv_path = tmp_path / "test_aro_index.tsv"
    fasta_path.write_text("\n".join(fasta_lines) + "\n")
    tsv_path.write_text("\n".join(tsv_lines) + "\n")
    return fasta_path, tsv_path


def _make_config(
    fasta_path: Path, aro_index_path: Path, output_dir: Path, injection_mode: str
) -> dict[str, Any]:
    """Small synthetic config: 8M model, CPU, tiny classifier, 1 epoch."""
    return {
        "paths": {
            "card_fasta": str(fasta_path),
            "aro_index": str(aro_index_path),
            "output_dir": str(output_dir),
        },
        "model": {
            "esm2_variant": MODEL_NAME,
            "device": "cpu",
            "injection_mode": injection_mode,
            "freeze_esm2": True,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 1.0e-3,
            "epochs": 1,
            "optimizer": "adam",
        },
        "classifier": {"hidden_dim": 16, "dropout": 0.1},
        "loss": {
            "weight_drug_class": 1.0,
            "weight_resistance_mechanism": 1.0,
            "weight_amr_gene_family": 1.0,
        },
        "logging": {"wandb_project": "test-project", "wandb_run_name": "test-run"},
    }


@pytest.fixture(scope="module")
def card_files(tmp_path_factory) -> tuple[Path, Path]:
    tmp_path = tmp_path_factory.mktemp("card_data")
    return _make_card_files(tmp_path)


# ---------------------------------------------------------------------------
# build_dataloaders_from_records
# ---------------------------------------------------------------------------


class TestBuildDataloadersFromRecords:
    def test_all_three_splits_non_empty(self, card_files):
        fasta_path, aro_index_path = card_files
        records = load_card_dataset(fasta_path, aro_index_path)
        train_loader, val_loader, test_loader, label_vocabularies = (
            build_dataloaders_from_records(records, batch_size=2)
        )
        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0
        assert len(test_loader.dataset) > 0

    def test_label_vocab_covers_all_records(self, card_files):
        """Vocab must be built from the full set, not just the train split."""
        fasta_path, aro_index_path = card_files
        records = load_card_dataset(fasta_path, aro_index_path)
        _, _, _, label_vocabularies = build_dataloaders_from_records(records, batch_size=2)
        assert len(label_vocabularies["resistance_mechanism"]) == 2


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def test_adam_returns_adam_optimizer(self):
        param = torch.nn.Parameter(torch.zeros(2))
        optimizer = build_optimizer("adam", [param], 1e-3)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_case_insensitive(self):
        param = torch.nn.Parameter(torch.zeros(2))
        optimizer = build_optimizer("Adam", [param], 1e-3)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_unsupported_optimizer_raises(self):
        param = torch.nn.Parameter(torch.zeros(2))
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            build_optimizer("sgd", [param], 1e-3)


# ---------------------------------------------------------------------------
# run_epoch
# ---------------------------------------------------------------------------


class TestRunEpoch:
    @pytest.fixture()
    def wired_models(self, card_files):
        fasta_path, aro_index_path = card_files
        records = load_card_dataset(fasta_path, aro_index_path)
        train_loader, _, _, label_vocabularies = build_dataloaders_from_records(
            records, batch_size=2
        )
        config = _make_config(fasta_path, aro_index_path, Path("unused"), "internal")
        esm2, soft_prompt, classifier, loss_fn = build_models(config, label_vocabularies, "cpu")
        return train_loader, esm2, soft_prompt, classifier, loss_fn

    def test_train_epoch_returns_finite_metrics(self, wired_models):
        train_loader, esm2, soft_prompt, classifier, loss_fn = wired_models
        optimizer = build_optimizer("adam", list(soft_prompt.parameters()) + list(
            classifier.parameters()
        ), 1e-3)
        metrics = run_epoch(train_loader, esm2, soft_prompt, classifier, loss_fn, "cpu", optimizer)
        expected_keys = {
            "total", "drug_class", "resistance_mechanism", "amr_gene_family",
            "resistance_mechanism_accuracy", "amr_gene_family_accuracy", "drug_class_f1_micro",
        }
        assert set(metrics.keys()) == expected_keys
        for value in metrics.values():
            assert torch.isfinite(torch.tensor(value))

    def test_training_epoch_updates_trainable_params(self, wired_models):
        train_loader, esm2, soft_prompt, classifier, loss_fn = wired_models
        params_before = [p.clone() for p in soft_prompt.parameters()]
        optimizer = build_optimizer("adam", list(soft_prompt.parameters()) + list(
            classifier.parameters()
        ), 1e-2)
        run_epoch(train_loader, esm2, soft_prompt, classifier, loss_fn, "cpu", optimizer)
        params_after = list(soft_prompt.parameters())
        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )

    def test_eval_epoch_does_not_update_params(self, wired_models):
        train_loader, esm2, soft_prompt, classifier, loss_fn = wired_models
        params_before = [p.clone() for p in soft_prompt.parameters()]
        run_epoch(train_loader, esm2, soft_prompt, classifier, loss_fn, "cpu", optimizer=None)
        params_after = list(soft_prompt.parameters())
        assert all(
            torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )


# ---------------------------------------------------------------------------
# Full train() integration
# ---------------------------------------------------------------------------


class TestTrainIntegration:
    @pytest.mark.parametrize("injection_mode", ["internal", "external"])
    def test_train_runs_and_writes_checkpoint(
        self, card_files, tmp_path, monkeypatch, injection_mode
    ):
        monkeypatch.setenv("WANDB_MODE", "disabled")
        fasta_path, aro_index_path = card_files
        output_dir = tmp_path / f"outputs_{injection_mode}"
        config = _make_config(fasta_path, aro_index_path, output_dir, injection_mode)

        train(config)

        checkpoint_path = output_dir / "best_model.pt"
        assert checkpoint_path.exists()

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert set(checkpoint.keys()) == {
            "epoch", "soft_prompt_state_dict", "classifier_state_dict", "best_val_loss",
        }
        assert torch.isfinite(torch.tensor(checkpoint["best_val_loss"]))
