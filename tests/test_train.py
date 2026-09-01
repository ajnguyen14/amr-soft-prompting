"""Smoke tests for the training loop (src/training/train.py).

Uses the 8M model on CPU with a small synthetic CARD-like dataset (tiny FASTA +
ARO index written to tmp_path), per CLAUDE.md's smoke-test convention. Never
touches the real CARD dataset or the 650M model -- that's a GPU-server-only
training run, out of scope for a CPU smoke test.
"""

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest
import torch

from src.training.train import (
    build_dataloaders,
    build_dataloaders_from_records,
    build_models,
    build_optimizer,
    build_v2_models,
    run_epoch,
    run_v2_epoch,
    train,
    train_v2,
)
from src.data.card_parser import load_card_dataset
from src.models.soft_prompt import NullSoftPrompt

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
        "loss": {"weight_amr_gene_family": 1.0},
        "logging": {"wandb_project": "test-project", "wandb_run_name": "test-run"},
    }


def _make_v2_config(
    fasta_path: Path,
    aro_index_path: Path,
    output_dir: Path,
    injection_mode: str,
    conditioning_field: str,
    target_field: str,
    ta_proximity_results_path: Path | None = None,
) -> dict[str, Any]:
    """Small synthetic V2 config: 8M model, CPU, tiny classifier, 1 epoch."""
    config = _make_config(fasta_path, aro_index_path, output_dir, injection_mode)
    if ta_proximity_results_path is not None:
        config["paths"]["ta_proximity_results"] = str(ta_proximity_results_path)
    config["loss"] = {"weight": 1.0}
    config["task"] = {"conditioning_field": conditioning_field, "target_field": target_field}
    return config


def _make_ta_proximity_file(tmp_path: Path, aro_accessions: list[str]) -> Path:
    """Write a ta_proximity_results.json-shaped fixture (Run 3's conditioning input).

    Cycles deterministically through the 3-way collapsed categorical so every
    category is exercised in a small fixture, mirroring the real
    scripts/run_ta_proximity.py output shape (a list of dicts with at least
    'aro_accession' and 'category').
    """
    categories = ["distance", "no_ta_locus", "unknown"]
    entries = [
        {"aro_accession": aro, "category": categories[i % 3], "distance_bp": None}
        for i, aro in enumerate(aro_accessions)
    ]
    path = tmp_path / "ta_proximity_results.json"
    path.write_text(json.dumps(entries))
    return path


@pytest.fixture(scope="module")
def card_files(tmp_path_factory) -> tuple[Path, Path]:
    tmp_path = tmp_path_factory.mktemp("card_data")
    return _make_card_files(tmp_path)


@pytest.fixture(scope="module")
def ta_proximity_file(card_files, tmp_path_factory) -> Path:
    """ta_proximity_results.json fixture covering every ARO accession in card_files."""
    fasta_path, aro_index_path = card_files
    records = load_card_dataset(fasta_path, aro_index_path)
    tmp_path = tmp_path_factory.mktemp("ta_proximity_data")
    return _make_ta_proximity_file(tmp_path, [r.aro_accession for r in records])


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
        expected_keys = {"total", "amr_gene_family", "amr_gene_family_accuracy"}
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


# ---------------------------------------------------------------------------
# build_v2_models
# ---------------------------------------------------------------------------


class TestBuildV2Models:
    def test_ce_target_wires_correct_shapes(self, card_files):
        """conditioning_field='amr_gene_family', target_field='resistance_mechanism' (Run 2 shape)."""
        fasta_path, aro_index_path = card_files
        records = load_card_dataset(fasta_path, aro_index_path)
        _, _, _, label_vocabularies = build_dataloaders_from_records(records, batch_size=2)
        config = _make_v2_config(
            fasta_path, aro_index_path, Path("unused"), "internal",
            "amr_gene_family", "resistance_mechanism",
        )
        esm2, soft_prompt, classifier, loss_fn = build_v2_models(config, label_vocabularies, "cpu")

        assert soft_prompt.embedding.num_embeddings == len(label_vocabularies["amr_gene_family"])
        assert classifier.head.out_features == len(label_vocabularies["resistance_mechanism"])
        assert classifier.target_name == "resistance_mechanism"
        assert loss_fn.loss_type == "ce"
        assert loss_fn.batch_key == "resistance_mechanism"

    def test_bce_target_wires_correct_shapes(self, card_files):
        """conditioning_field='amr_gene_family', target_field='drug_class' (Run 1 shape)."""
        fasta_path, aro_index_path = card_files
        records = load_card_dataset(fasta_path, aro_index_path)
        _, _, _, label_vocabularies = build_dataloaders_from_records(records, batch_size=2)
        config = _make_v2_config(
            fasta_path, aro_index_path, Path("unused"), "internal",
            "amr_gene_family", "drug_class",
        )
        esm2, soft_prompt, classifier, loss_fn = build_v2_models(config, label_vocabularies, "cpu")

        assert classifier.head.out_features == len(label_vocabularies["drug_class"])
        assert classifier.target_name == "drug_class"
        assert loss_fn.loss_type == "bce"
        assert loss_fn.batch_key == "drug_class_labels"

    def test_multilabel_conditioning_field_raises(self, card_files):
        """drug_class is multi-label -- can't feed SingleFieldSoftPrompt an index tensor."""
        fasta_path, aro_index_path = card_files
        records = load_card_dataset(fasta_path, aro_index_path)
        _, _, _, label_vocabularies = build_dataloaders_from_records(records, batch_size=2)
        config = _make_v2_config(
            fasta_path, aro_index_path, Path("unused"), "internal",
            "drug_class", "amr_gene_family",
        )
        with pytest.raises(ValueError, match="multi-label"):
            build_v2_models(config, label_vocabularies, "cpu")

    def test_ta_proximity_conditioning_wires_correct_shapes(self, card_files, ta_proximity_file):
        """conditioning_field='ta_proximity', target_field='amr_gene_family' (Run 3 shape).

        ta_proximity is the collapsed 3-way categorical (CLAUDE.md's
        sparse-signal decision) -- SingleFieldSoftPrompt's embedding table
        must have exactly 3 rows, one per category.
        """
        fasta_path, aro_index_path = card_files
        config = _make_v2_config(
            fasta_path, aro_index_path, Path("unused"), "internal",
            "ta_proximity", "amr_gene_family",
            ta_proximity_results_path=ta_proximity_file,
        )
        _, _, _, label_vocabularies = build_dataloaders(config)
        esm2, soft_prompt, classifier, loss_fn = build_v2_models(config, label_vocabularies, "cpu")

        assert label_vocabularies["ta_proximity"] == ["distance", "no_ta_locus", "unknown"]
        assert soft_prompt.embedding.num_embeddings == 3
        assert classifier.head.out_features == len(label_vocabularies["amr_gene_family"])
        assert classifier.target_name == "amr_gene_family"
        assert loss_fn.loss_type == "ce"
        assert loss_fn.batch_key == "amr_gene_family"

    def test_none_conditioning_wires_null_soft_prompt(self, card_files):
        """conditioning_field='none', target_field='amr_gene_family' (negative control, pairs with Run 3).

        No real conditioning field is read at all -- build_v2_models must
        produce a NullSoftPrompt (fixed zero token, no embedding table) rather
        than looking 'none' up in TARGET_FIELD_SPECS/label_vocabularies.
        """
        fasta_path, aro_index_path = card_files
        records = load_card_dataset(fasta_path, aro_index_path)
        _, _, _, label_vocabularies = build_dataloaders_from_records(records, batch_size=2)
        config = _make_v2_config(
            fasta_path, aro_index_path, Path("unused"), "internal",
            "none", "amr_gene_family",
        )
        esm2, soft_prompt, classifier, loss_fn = build_v2_models(config, label_vocabularies, "cpu")

        assert isinstance(soft_prompt, NullSoftPrompt)
        assert len(list(soft_prompt.parameters())) == 0
        assert classifier.head.out_features == len(label_vocabularies["amr_gene_family"])
        assert classifier.target_name == "amr_gene_family"
        assert loss_fn.loss_type == "ce"
        assert loss_fn.batch_key == "amr_gene_family"


# ---------------------------------------------------------------------------
# run_v2_epoch
# ---------------------------------------------------------------------------


class TestRunV2Epoch:
    @pytest.fixture()
    def wired_v2_models(self, card_files):
        fasta_path, aro_index_path = card_files
        records = load_card_dataset(fasta_path, aro_index_path)
        train_loader, _, _, label_vocabularies = build_dataloaders_from_records(
            records, batch_size=2
        )
        config = _make_v2_config(
            fasta_path, aro_index_path, Path("unused"), "internal",
            "amr_gene_family", "resistance_mechanism",
        )
        esm2, soft_prompt, classifier, loss_fn = build_v2_models(config, label_vocabularies, "cpu")
        return train_loader, esm2, soft_prompt, classifier, loss_fn

    def test_train_epoch_returns_finite_metrics(self, wired_v2_models):
        train_loader, esm2, soft_prompt, classifier, loss_fn = wired_v2_models
        optimizer = build_optimizer(
            "adam", list(soft_prompt.parameters()) + list(classifier.parameters()), 1e-3
        )
        metrics = run_v2_epoch(
            train_loader, esm2, soft_prompt, classifier, loss_fn, "amr_gene_family", "cpu", optimizer
        )
        expected_keys = {"total", "resistance_mechanism", "resistance_mechanism_accuracy"}
        assert set(metrics.keys()) == expected_keys
        for value in metrics.values():
            assert torch.isfinite(torch.tensor(value))

    def test_training_epoch_updates_trainable_params(self, wired_v2_models):
        train_loader, esm2, soft_prompt, classifier, loss_fn = wired_v2_models
        params_before = [p.clone() for p in soft_prompt.parameters()]
        optimizer = build_optimizer(
            "adam", list(soft_prompt.parameters()) + list(classifier.parameters()), 1e-2
        )
        run_v2_epoch(
            train_loader, esm2, soft_prompt, classifier, loss_fn, "amr_gene_family", "cpu", optimizer
        )
        params_after = list(soft_prompt.parameters())
        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )

    def test_eval_epoch_does_not_update_params(self, wired_v2_models):
        train_loader, esm2, soft_prompt, classifier, loss_fn = wired_v2_models
        params_before = [p.clone() for p in soft_prompt.parameters()]
        run_v2_epoch(
            train_loader, esm2, soft_prompt, classifier, loss_fn, "amr_gene_family", "cpu",
            optimizer=None,
        )
        params_after = list(soft_prompt.parameters())
        assert all(
            torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )

    def test_none_conditioning_runs_end_to_end(self, card_files):
        """run_v2_epoch's conditioning_batch_key=None branch (negative control)."""
        fasta_path, aro_index_path = card_files
        records = load_card_dataset(fasta_path, aro_index_path)
        train_loader, _, _, label_vocabularies = build_dataloaders_from_records(
            records, batch_size=2
        )
        config = _make_v2_config(
            fasta_path, aro_index_path, Path("unused"), "internal",
            "none", "amr_gene_family",
        )
        esm2, soft_prompt, classifier, loss_fn = build_v2_models(config, label_vocabularies, "cpu")
        optimizer = build_optimizer(
            "adam", list(soft_prompt.parameters()) + list(classifier.parameters()), 1e-3
        )
        metrics = run_v2_epoch(
            train_loader, esm2, soft_prompt, classifier, loss_fn, "none", "cpu", optimizer
        )
        expected_keys = {"total", "amr_gene_family", "amr_gene_family_accuracy"}
        assert set(metrics.keys()) == expected_keys
        for value in metrics.values():
            assert torch.isfinite(torch.tensor(value))


# ---------------------------------------------------------------------------
# Full train_v2() integration
# ---------------------------------------------------------------------------


class TestTrainV2Integration:
    @pytest.mark.parametrize(
        "conditioning_field,target_field",
        [
            ("amr_gene_family", "resistance_mechanism"),  # Run 2 shape, loss_type='ce'
            ("amr_gene_family", "drug_class"),  # Run 1 shape, loss_type='bce'
            ("ta_proximity", "amr_gene_family"),  # Run 3 shape, collapsed 3-way categorical
        ],
    )
    def test_train_v2_runs_and_writes_checkpoint(
        self, card_files, ta_proximity_file, tmp_path, monkeypatch, conditioning_field, target_field
    ):
        monkeypatch.setenv("WANDB_MODE", "disabled")
        fasta_path, aro_index_path = card_files
        output_dir = tmp_path / f"outputs_v2_{target_field}_{conditioning_field}"
        ta_proximity_results_path = (
            ta_proximity_file if conditioning_field == "ta_proximity" else None
        )
        config = _make_v2_config(
            fasta_path, aro_index_path, output_dir, "internal", conditioning_field, target_field,
            ta_proximity_results_path=ta_proximity_results_path,
        )

        train_v2(config)

        checkpoint_path = output_dir / "best_model.pt"
        assert checkpoint_path.exists()

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert set(checkpoint.keys()) == {
            "epoch", "soft_prompt_state_dict", "classifier_state_dict", "best_val_loss",
        }
        assert torch.isfinite(torch.tensor(checkpoint["best_val_loss"]))

    def test_main_dispatches_to_train_v2_when_task_section_present(
        self, card_files, tmp_path, monkeypatch
    ):
        """main()'s dispatch logic: a config file with a 'task' section runs the V2 path
        end-to-end through the actual --config CLI entry point, not just build_v2_models
        called directly.
        """
        import yaml

        from src.training.train import main

        monkeypatch.setenv("WANDB_MODE", "disabled")
        fasta_path, aro_index_path = card_files
        output_dir = tmp_path / "outputs_main_dispatch"

        config = _make_v2_config(
            fasta_path, aro_index_path, output_dir, "internal",
            "amr_gene_family", "resistance_mechanism",
        )
        # load_config resolves base.yaml as config_path's sibling, so the
        # override file needs its own directory with an (empty) base.yaml.
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        (configs_dir / "base.yaml").write_text("{}\n")
        config_path = configs_dir / "task_override.yaml"
        config_path.write_text(yaml.dump(config))

        monkeypatch.setattr("sys.argv", ["run_training.py", "--config", str(config_path)])
        main()

        checkpoint_path = output_dir / "best_model.pt"
        assert checkpoint_path.exists()
