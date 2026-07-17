"""Smoke tests for the holdout evaluation script (src/eval/evaluate.py).

Uses the 8M model on CPU with a small synthetic CARD-like dataset (tiny FASTA +
ARO index written to tmp_path), per CLAUDE.md's smoke-test convention. Never
touches the real CARD dataset or the 650M model.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from src.eval.evaluate import evaluate, top_confused_pairs
from src.data.card_parser import load_card_dataset
from src.training.train import train

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
NUM_ACCESSIONS_PER_MECHANISM = 10  # -> 8/1/1 train/val/test split per mechanism


def _make_card_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write a small synthetic FASTA + ARO index: 2 mechanisms x 10 accessions each."""
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
    fasta_path: Path, aro_index_path: Path, output_dir: Path, injection_mode: str = "internal"
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


@pytest.fixture(scope="module")
def card_files(tmp_path_factory) -> tuple[Path, Path]:
    tmp_path = tmp_path_factory.mktemp("card_data")
    return _make_card_files(tmp_path)


@pytest.fixture(scope="module")
def trained_checkpoint(card_files, tmp_path_factory, monkeypatch_module) -> tuple[dict, Path]:
    fasta_path, aro_index_path = card_files
    output_dir = tmp_path_factory.mktemp("outputs")
    config = _make_config(fasta_path, aro_index_path, output_dir)
    train(config)
    return config, output_dir / "best_model.pt"


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch so WANDB_MODE can be set once for the module-scoped checkpoint fixture."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    mp.setenv("WANDB_MODE", "disabled")
    yield mp
    mp.undo()


class TestEvaluate:
    def test_writes_results_json_with_expected_top_level_keys(self, trained_checkpoint):
        config, checkpoint_path = trained_checkpoint
        results = evaluate(config, checkpoint_path)

        assert set(results.keys()) == {
            "checkpoint", "checkpoint_epoch", "eval_dir", "config", "timestamp",
            "aggregate", "amr_gene_family",
        }

        results_json_path = Path(results["eval_dir"]) / "evaluation_results.json"
        assert results_json_path.exists()
        with open(results_json_path) as f:
            on_disk = json.load(f)
        assert on_disk["checkpoint"] == results["checkpoint"]

    def test_amr_gene_family_reports_aggregate_and_csv_not_plot(self, trained_checkpoint):
        config, checkpoint_path = trained_checkpoint
        results = evaluate(config, checkpoint_path)

        family_results = results["amr_gene_family"]
        assert 0.0 <= family_results["accuracy"] <= 1.0
        assert 0.0 <= family_results["macro_f1"] <= 1.0
        assert isinstance(family_results["top_confused_pairs"], list)

        csv_path = Path(family_results["confusion_matrix_csv"])
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"
        # No PNG should be written for the 398-class task.
        assert not (Path(results["eval_dir"]) / "confusion_matrix_amr_gene_family.png").exists()

    def test_aggregate_matches_compute_metrics_keys(self, trained_checkpoint):
        config, checkpoint_path = trained_checkpoint
        results = evaluate(config, checkpoint_path)

        assert set(results["aggregate"].keys()) == {"amr_gene_family_accuracy"}


class TestTopConfusedPairs:
    def test_excludes_diagonal_and_sorts_descending(self):
        import numpy as np

        cm = np.array([[5, 3, 0], [1, 4, 2], [0, 0, 6]])
        labels = ["a", "b", "c"]
        pairs = top_confused_pairs(cm, labels, top_n=10)

        assert {"true": "a", "predicted": "b", "count": 3} in pairs
        assert not any(p["true"] == p["predicted"] for p in pairs)
        counts = [p["count"] for p in pairs]
        assert counts == sorted(counts, reverse=True)

    def test_respects_top_n(self):
        import numpy as np

        cm = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        labels = ["a", "b", "c"]
        pairs = top_confused_pairs(cm, labels, top_n=2)
        assert len(pairs) == 2
