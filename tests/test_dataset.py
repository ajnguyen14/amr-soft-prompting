"""Smoke tests for AMRDataset (src/data/dataset.py)."""

import dataclasses
from pathlib import Path

import torch
import pytest

from src.data.card_parser import CARDRecord, get_label_vocabularies, load_card_dataset
from src.data.dataset import AMRDataset, split_dataset

# ---------------------------------------------------------------------------
# Paths to the real CARD data — tests requiring these are skipped if absent.
# ---------------------------------------------------------------------------
_CARD_DIR = Path(__file__).parent.parent / "data" / "raw"
_FASTA_PATH = _CARD_DIR / "protein_fasta_protein_homolog_model.fasta"
_ARO_INDEX_PATH = _CARD_DIR / "aro_index.tsv"

_REAL_DATA_AVAILABLE = _FASTA_PATH.exists() and _ARO_INDEX_PATH.exists()
_skip_no_data = pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason="CARD data files not present (expected on CPU/GPU server)",
)


# ---------------------------------------------------------------------------
# Fixture records — in-memory, no file I/O required.
# Three entries mirror the minimal pipeline fixture; Erm(34) is the
# maximum-drug-class case in this fixture (3 classes).
# ---------------------------------------------------------------------------

_RECORDS = [
    CARDRecord(
        aro_accession="ARO:3002999",
        protein_accession="ACT97415.1",
        gene_name="CblA-1",
        organism="mixed culture bacterium AX_gF3SD01_15",
        sequence="MKAYFIAILTLFTCIATVVRAQQMSELENRIDSLLNGK",
        drug_classes=["cephalosporin"],
        resistance_mechanism="antibiotic inactivation",
        amr_gene_family="CblA beta-lactamase",
        card_short_name="CblA-1",
    ),
    CARDRecord(
        aro_accession="ARO:3002524",
        protein_accession="AAC44793.1",
        gene_name="AAC(2')-Ib",
        organism="Providencia stuartii",
        sequence="MFGSKLSKTIAAFAALVSSAATMA",
        drug_classes=["aminoglycoside antibiotic"],
        resistance_mechanism="antibiotic inactivation",
        amr_gene_family="AAC(2')",
        card_short_name="AAC(2')-Ib",
    ),
    CARDRecord(
        aro_accession="ARO:3000600",
        protein_accession="AAP74657.1",
        gene_name="Erm(34)",
        organism="Alkalihalobacillus clausii",
        sequence="MTKKMNKYNGKKLSRGEPPNFSGQHFMHNKRLLKEIVDK",
        drug_classes=["lincosamide antibiotic", "macrolide antibiotic", "streptogramin antibiotic"],
        resistance_mechanism="antibiotic target alteration",
        amr_gene_family="Erm 23S ribosomal RNA methyltransferase",
        card_short_name="Erm(34)",
    ),
]


@pytest.fixture()
def dataset_and_vocabs() -> tuple[AMRDataset, dict[str, list[str]]]:
    """Return (AMRDataset, label_vocabularies) built from three in-memory records."""
    vocabs = get_label_vocabularies(_RECORDS)
    return AMRDataset(_RECORDS, vocabs), vocabs


# ---------------------------------------------------------------------------
# Tests: basic dataset properties
# ---------------------------------------------------------------------------


class TestAMRDatasetLength:
    def test_len_matches_record_count(self, dataset_and_vocabs):
        dataset, _ = dataset_and_vocabs
        assert len(dataset) == len(_RECORDS)


class TestAMRDatasetItemStructure:
    def test_item_has_expected_keys(self, dataset_and_vocabs):
        dataset, _ = dataset_and_vocabs
        item = dataset[0]
        assert set(item.keys()) == {
            "sequence",
            "drug_class_labels",
            "resistance_mechanism",
            "amr_gene_family",
            "aro_accession",
        }

    def test_sequence_is_non_empty_str(self, dataset_and_vocabs):
        dataset, _ = dataset_and_vocabs
        for i in range(len(dataset)):
            seq = dataset[i]["sequence"]
            assert isinstance(seq, str) and len(seq) > 0

    def test_sequence_matches_record(self, dataset_and_vocabs):
        dataset, _ = dataset_and_vocabs
        for i, record in enumerate(_RECORDS):
            assert dataset[i]["sequence"] == record.sequence

    def test_aro_accession_passthrough(self, dataset_and_vocabs):
        dataset, _ = dataset_and_vocabs
        for i, record in enumerate(_RECORDS):
            assert dataset[i]["aro_accession"] == record.aro_accession

    def test_drug_class_labels_is_float32_tensor(self, dataset_and_vocabs):
        dataset, _ = dataset_and_vocabs
        for i in range(len(dataset)):
            t = dataset[i]["drug_class_labels"]
            assert isinstance(t, torch.Tensor)
            assert t.dtype == torch.float32

    def test_drug_class_labels_shape(self, dataset_and_vocabs):
        dataset, vocabs = dataset_and_vocabs
        expected = (len(vocabs["drug_class"]),)
        for i in range(len(dataset)):
            assert dataset[i]["drug_class_labels"].shape == expected

    def test_resistance_mechanism_is_long_scalar(self, dataset_and_vocabs):
        dataset, _ = dataset_and_vocabs
        for i in range(len(dataset)):
            t = dataset[i]["resistance_mechanism"]
            assert isinstance(t, torch.Tensor)
            assert t.dtype == torch.long
            assert t.shape == ()

    def test_amr_gene_family_is_long_scalar(self, dataset_and_vocabs):
        dataset, _ = dataset_and_vocabs
        for i in range(len(dataset)):
            t = dataset[i]["amr_gene_family"]
            assert isinstance(t, torch.Tensor)
            assert t.dtype == torch.long
            assert t.shape == ()


# ---------------------------------------------------------------------------
# Tests: multi-hot drug class encoding
# ---------------------------------------------------------------------------


class TestMultiHotDrugClass:
    def test_single_class_sets_exactly_one_bit(self, dataset_and_vocabs):
        """CblA-1 has one drug class — exactly one bit should be hot."""
        dataset, vocabs = dataset_and_vocabs
        item = dataset[0]  # CblA-1: drug_classes=["cephalosporin"]
        assert item["drug_class_labels"].sum().item() == pytest.approx(1.0)
        ceph_idx = vocabs["drug_class"].index("cephalosporin")
        assert item["drug_class_labels"][ceph_idx].item() == pytest.approx(1.0)

    def test_multi_hot_sets_correct_bits(self, dataset_and_vocabs):
        """Erm(34) has three drug classes — exactly those three bits should be hot."""
        dataset, vocabs = dataset_and_vocabs
        item = dataset[2]  # Erm(34): 3 drug classes
        dc_vocab = vocabs["drug_class"]
        assert item["drug_class_labels"].sum().item() == pytest.approx(3.0)
        for label in ["lincosamide antibiotic", "macrolide antibiotic", "streptogramin antibiotic"]:
            idx = dc_vocab.index(label)
            assert item["drug_class_labels"][idx].item() == pytest.approx(1.0), (
                f"Expected bit set for '{label}' at index {idx}"
            )

    def test_non_present_classes_are_zero(self, dataset_and_vocabs):
        """Bits for drug classes absent from a record must be zero."""
        dataset, vocabs = dataset_and_vocabs
        item = dataset[0]  # CblA-1: only cephalosporin
        dc_vocab = vocabs["drug_class"]
        for label in ["aminoglycoside antibiotic", "lincosamide antibiotic",
                      "macrolide antibiotic", "streptogramin antibiotic"]:
            idx = dc_vocab.index(label)
            assert item["drug_class_labels"][idx].item() == pytest.approx(0.0), (
                f"Expected bit zero for '{label}' at index {idx}"
            )

    def test_max_drug_classes_in_fixture(self, dataset_and_vocabs):
        """Edge case: Erm(34) is the maximum-drug-class record in this fixture (3 classes)."""
        dataset, _ = dataset_and_vocabs
        item = dataset[2]  # Erm(34)
        vec = item["drug_class_labels"]
        # All values must be exactly 0 or 1 — no intermediate values
        assert torch.all((vec == 0.0) | (vec == 1.0))
        assert int(vec.sum().item()) == 3


# ---------------------------------------------------------------------------
# Tests: single-index labels (resistance_mechanism and amr_gene_family)
# ---------------------------------------------------------------------------


class TestSingleIndexLabels:
    def test_resistance_mechanism_index_in_range(self, dataset_and_vocabs):
        dataset, vocabs = dataset_and_vocabs
        vocab_size = len(vocabs["resistance_mechanism"])
        for i in range(len(dataset)):
            idx = dataset[i]["resistance_mechanism"].item()
            assert 0 <= idx < vocab_size

    def test_amr_gene_family_index_in_range(self, dataset_and_vocabs):
        dataset, vocabs = dataset_and_vocabs
        vocab_size = len(vocabs["amr_gene_family"])
        for i in range(len(dataset)):
            idx = dataset[i]["amr_gene_family"].item()
            assert 0 <= idx < vocab_size

    def test_resistance_mechanism_correct_index(self, dataset_and_vocabs):
        """Each record's mechanism must map to the correct vocabulary index."""
        dataset, vocabs = dataset_and_vocabs
        mech_vocab = vocabs["resistance_mechanism"]
        for i, record in enumerate(_RECORDS):
            expected = mech_vocab.index(record.resistance_mechanism)
            actual = dataset[i]["resistance_mechanism"].item()
            assert actual == expected, (
                f"Record {i} ({record.gene_name}): expected mechanism index "
                f"{expected} ('{record.resistance_mechanism}'), got {actual}"
            )

    def test_amr_gene_family_correct_index(self, dataset_and_vocabs):
        """Each record's gene family must map to the correct vocabulary index."""
        dataset, vocabs = dataset_and_vocabs
        family_vocab = vocabs["amr_gene_family"]
        for i, record in enumerate(_RECORDS):
            expected = family_vocab.index(record.amr_gene_family)
            actual = dataset[i]["amr_gene_family"].item()
            assert actual == expected, (
                f"Record {i} ({record.gene_name}): expected family index "
                f"{expected} ('{record.amr_gene_family}'), got {actual}"
            )

    def test_different_mechanisms_get_different_indices(self, dataset_and_vocabs):
        """CblA-1 (inactivation) and Erm(34) (target alteration) must get distinct indices."""
        dataset, _ = dataset_and_vocabs
        cbl_mech = dataset[0]["resistance_mechanism"].item()
        erm_mech = dataset[2]["resistance_mechanism"].item()
        assert cbl_mech != erm_mech


# ---------------------------------------------------------------------------
# Tests: ta_proximity conditioning field (Run 3, collapsed 3-way categorical)
# ---------------------------------------------------------------------------

_TA_PROXIMITY_RECORDS = [
    dataclasses.replace(_RECORDS[0], ta_proximity_category="distance"),
    dataclasses.replace(_RECORDS[1], ta_proximity_category="no_ta_locus"),
    dataclasses.replace(_RECORDS[2], ta_proximity_category="unknown"),
]


@pytest.fixture()
def ta_proximity_dataset_and_vocabs() -> tuple[AMRDataset, dict[str, list[str]]]:
    vocabs = get_label_vocabularies(_TA_PROXIMITY_RECORDS)
    return AMRDataset(_TA_PROXIMITY_RECORDS, vocabs), vocabs


class TestTAProximityConditioningField:
    def test_ta_proximity_absent_when_vocab_not_loaded(self, dataset_and_vocabs):
        """Run 1/2 datasets (no ta_proximity_category set) must not emit this key."""
        dataset, _ = dataset_and_vocabs
        assert "ta_proximity" not in dataset[0]

    def test_ta_proximity_present_when_vocab_loaded(self, ta_proximity_dataset_and_vocabs):
        dataset, _ = ta_proximity_dataset_and_vocabs
        for i in range(len(dataset)):
            assert "ta_proximity" in dataset[i]

    def test_ta_proximity_is_long_scalar(self, ta_proximity_dataset_and_vocabs):
        dataset, _ = ta_proximity_dataset_and_vocabs
        for i in range(len(dataset)):
            t = dataset[i]["ta_proximity"]
            assert isinstance(t, torch.Tensor)
            assert t.dtype == torch.long
            assert t.shape == ()

    def test_ta_proximity_correct_index(self, ta_proximity_dataset_and_vocabs):
        dataset, vocabs = ta_proximity_dataset_and_vocabs
        ta_vocab = vocabs["ta_proximity"]
        for i, record in enumerate(_TA_PROXIMITY_RECORDS):
            expected = ta_vocab.index(record.ta_proximity_category)
            actual = dataset[i]["ta_proximity"].item()
            assert actual == expected

    def test_vocab_is_exactly_the_three_way_categorical(self, ta_proximity_dataset_and_vocabs):
        _, vocabs = ta_proximity_dataset_and_vocabs
        assert vocabs["ta_proximity"] == ["distance", "no_ta_locus", "unknown"]


# ---------------------------------------------------------------------------
# Unit tests: train/val/test split
# ---------------------------------------------------------------------------


def _make_split_record(aro_accession: str, mechanism: str, protein_accession: str) -> CARDRecord:
    """Build a minimal synthetic CARDRecord for split-logic tests."""
    return CARDRecord(
        aro_accession=aro_accession,
        protein_accession=protein_accession,
        gene_name="gene",
        organism="organism",
        sequence="MKAYFIAILT",
        drug_classes=["some antibiotic"],
        resistance_mechanism=mechanism,
        amr_gene_family="some family",
        card_short_name="short",
    )


@pytest.fixture()
def stratified_records() -> list[CARDRecord]:
    """10 ARO accessions per mechanism across 2 mechanisms (20 total accessions).

    One accession (ARO:A000) has two sequence records, to exercise the
    multi-sequence-per-accession grouping guarantee. 21 records, 20 accessions.
    """
    records = []
    for i in range(10):
        aro = f"ARO:A{i:03d}"
        records.append(_make_split_record(aro, "mechanism_A", f"protA{i}.1"))
        if i == 0:
            # Second sequence variant under the same accession.
            records.append(_make_split_record(aro, "mechanism_A", f"protA{i}.2"))
    for i in range(10):
        aro = f"ARO:B{i:03d}"
        records.append(_make_split_record(aro, "mechanism_B", f"protB{i}.1"))
    return records


class TestSplitDataset:
    def test_all_records_included_exactly_once(self, stratified_records):
        splits = split_dataset(stratified_records, seed=42)
        combined = splits["train"] + splits["val"] + splits["test"]
        assert len(combined) == len(stratified_records)
        assert {r.protein_accession for r in combined} == {
            r.protein_accession for r in stratified_records
        }

    def test_no_accession_split_across_sets(self, stratified_records):
        """Every record for a given ARO accession must land in the same split."""
        splits = split_dataset(stratified_records, seed=42)
        accession_to_splits: dict[str, set[str]] = {}
        for split_name in ("train", "val", "test"):
            for record in splits[split_name]:
                accession_to_splits.setdefault(record.aro_accession, set()).add(split_name)
        for aro_accession, split_names in accession_to_splits.items():
            assert len(split_names) == 1, (
                f"{aro_accession} appears in multiple splits: {split_names}"
            )

    def test_multi_sequence_accession_stays_together(self, stratified_records):
        splits = split_dataset(stratified_records, seed=42)
        for split_name in ("train", "val", "test"):
            protein_accs = {
                r.protein_accession for r in splits[split_name] if r.aro_accession == "ARO:A000"
            }
            if protein_accs:
                assert protein_accs == {"protA0.1", "protA0.2"}

    def test_stratified_ratio_per_mechanism(self, stratified_records):
        # 10 accessions per mechanism, default 80/10/10 -> 8/1/1 each, exactly.
        splits = split_dataset(stratified_records, seed=42)
        for split_name, expected_per_mechanism in (("train", 8), ("val", 1), ("test", 1)):
            accessions = {r.aro_accession for r in splits[split_name]}
            mechanism_a = [a for a in accessions if a.startswith("ARO:A")]
            mechanism_b = [a for a in accessions if a.startswith("ARO:B")]
            assert len(mechanism_a) == expected_per_mechanism
            assert len(mechanism_b) == expected_per_mechanism

    def test_deterministic_given_same_seed(self, stratified_records):
        splits_1 = split_dataset(stratified_records, seed=42)
        splits_2 = split_dataset(stratified_records, seed=42)
        for split_name in ("train", "val", "test"):
            accs_1 = [r.protein_accession for r in splits_1[split_name]]
            accs_2 = [r.protein_accession for r in splits_2[split_name]]
            assert accs_1 == accs_2

    def test_invalid_fractions_raise(self, stratified_records):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            split_dataset(stratified_records, train_frac=0.8, val_frac=0.1, test_frac=0.2)

    def test_inconsistent_mechanism_raises(self):
        conflicting = [
            _make_split_record("ARO:9999", "mechanism_A", "prot1"),
            _make_split_record("ARO:9999", "mechanism_B", "prot2"),
        ]
        with pytest.raises(ValueError, match="inconsistent resistance_mechanism"):
            split_dataset(conflicting)

    def test_global_rng_state_untouched(self, stratified_records):
        """split_dataset must not perturb the global random module's state."""
        import random as random_module

        random_module.seed(1234)
        state_before = random_module.getstate()
        split_dataset(stratified_records, seed=42)
        assert random_module.getstate() == state_before


# ---------------------------------------------------------------------------
# Integration tests: split against the full CARD dataset (skipped when absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_dataset():
    if not _REAL_DATA_AVAILABLE:
        pytest.skip("CARD data files not present (expected on CPU/GPU server)")
    return load_card_dataset(_FASTA_PATH, _ARO_INDEX_PATH)


@_skip_no_data
class TestSplitDatasetFullCARD:
    def test_split_covers_all_records_with_no_overlap(self, full_dataset):
        splits = split_dataset(full_dataset, seed=42)
        combined_accessions = (
            {r.aro_accession for r in splits["train"]}
            | {r.aro_accession for r in splits["val"]}
            | {r.aro_accession for r in splits["test"]}
        )
        all_accessions = {r.aro_accession for r in full_dataset}
        assert combined_accessions == all_accessions

        train_accs = {r.aro_accession for r in splits["train"]}
        val_accs = {r.aro_accession for r in splits["val"]}
        test_accs = {r.aro_accession for r in splits["test"]}
        assert train_accs.isdisjoint(val_accs)
        assert train_accs.isdisjoint(test_accs)
        assert val_accs.isdisjoint(test_accs)

    def test_split_ratio_roughly_80_10_10(self, full_dataset):
        splits = split_dataset(full_dataset, seed=42)
        total = sum(len(splits[s]) for s in ("train", "val", "test"))
        train_frac = len(splits["train"]) / total
        val_frac = len(splits["val"]) / total
        test_frac = len(splits["test"]) / total
        # Per-mechanism rounding means this is approximate, not exact.
        assert 0.75 <= train_frac <= 0.85
        assert 0.05 <= val_frac <= 0.15
        assert 0.05 <= test_frac <= 0.15
