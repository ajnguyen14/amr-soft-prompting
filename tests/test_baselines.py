"""Smoke tests for src/eval/baselines.py (pure CPU, no ESM-2)."""

import numpy as np

from src.data.card_parser import CARDRecord
from src.eval.baselines import run_baselines, stratify_by_similarity


def _record(aro: str, seq: str, family: str, mech: str, drugs: list[str]) -> CARDRecord:
    return CARDRecord(
        aro_accession=aro,
        protein_accession=f"P{aro}",
        gene_name=aro,
        organism="x",
        sequence=seq,
        drug_classes=drugs,
        resistance_mechanism=mech,
        amr_gene_family=family,
        card_short_name=aro,
        ta_proximity_category="unknown",
    )


def _splits() -> dict[str, list[CARDRecord]]:
    seq_a, seq_b = "MKAYFIAILTLFTCIATVVRAQQ" * 2, "WWGGHHPPCCDDEEFFYYNNQQ" * 2
    train = [
        _record("1", seq_a, "famA", "efflux", ["d1"]),
        _record("2", seq_b, "famB", "inact", ["d2", "d3"]),
    ]
    test = [_record("3", seq_a, "famA", "efflux", ["d1"])]
    return {"train": train, "val": test, "test": test}


VOCABS = {
    "amr_gene_family": ["famA", "famB"],
    "resistance_mechanism": ["efflux", "inact"],
    "drug_class": ["d1", "d2", "d3"],
}


def test_single_label_baselines_run() -> None:
    results = run_baselines(_splits(), VOCABS, "amr_gene_family", "resistance_mechanism")
    assert results["nearest_neighbor"]["accuracy"] == 1.0
    assert results["conditioning_lookup"]["accuracy"] == 1.0
    assert 0.0 <= results["majority"]["accuracy"] <= 1.0
    assert sum(row["n"] for row in results["nearest_neighbor_by_similarity"]) == 1


def test_multi_label_baselines_and_null_conditioning() -> None:
    results = run_baselines(_splits(), VOCABS, "none", "drug_class")
    assert "conditioning_lookup" not in results
    assert results["nearest_neighbor"]["subset_accuracy"] == 1.0


def test_stratify_by_similarity_bins() -> None:
    rows = stratify_by_similarity(np.array([True, False]), np.array([0.3, 0.99]))
    assert rows[0]["n"] == 1 and rows[0]["accuracy"] == 1.0
    assert rows[-1]["n"] == 1 and rows[-1]["accuracy"] == 0.0
