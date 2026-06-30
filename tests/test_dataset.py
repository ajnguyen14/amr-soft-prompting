"""Smoke tests for AMRDataset (src/data/dataset.py)."""

import torch
import pytest

from src.data.card_parser import CARDRecord, get_label_vocabularies
from src.data.dataset import AMRDataset


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
