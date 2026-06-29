"""Smoke tests for the CARD data pipeline (card_parser.py)."""

import textwrap
from pathlib import Path

import pytest

from src.data.card_parser import (
    CARDRecord,
    _parse_fasta_header,
    get_label_vocabularies,
    load_card_dataset,
)

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
# Fixture: minimal in-memory FASTA + ARO index for header parsing unit tests
# ---------------------------------------------------------------------------

MINIMAL_FASTA = textwrap.dedent("""\
    >gb|ACT97415.1|ARO:3002999|CblA-1 [mixed culture bacterium AX_gF3SD01_15]
    MKAYFIAILTLFTCIATVVRAQQMSELENRIDSLLNGK
    >gb|AAC44793.1|ARO:3002524|AAC(2')-Ib [Providencia stuartii]
    MFGSKLSKTIAAFAALVSSAATMA
    >gb|AAP74657.1|ARO:3000600|Erm(34) [Alkalihalobacillus clausii]
    MTKKMNKYNGKKLSRGEPPNFSGQHFMHNKRLLKEIVDK
""")

MINIMAL_ARO_TSV = textwrap.dedent("""\
    ARO Accession\tCVTERM ID\tModel Sequence ID\tModel ID\tModel Name\tARO Name\tProtein Accession\tDNA Accession\tAMR Gene Family\tDrug Class\tResistance Mechanism\tCARD Short Name
    ARO:3002999\t39432\t6143\t3831\tCblA-1\tCblA-1\tACT97415.1\tGU256745.1\tCblA beta-lactamase\tcephalosporin\tantibiotic inactivation\tCblA-1
    ARO:3002524\t38924\t85\t746\tAAC(2')-Ib\tAAC(2')-Ib\tAAC44793.1\tU41471.1\tAAC(2')\taminoglycoside antibiotic\tantibiotic inactivation\tAAC(2')-Ib
    ARO:3000600\t37200\t4719\t1246\tErm(34)\tErm(34)\tAAP74657.1\tAY116765.1\tErm 23S ribosomal RNA methyltransferase\tlincosamide antibiotic;macrolide antibiotic;streptogramin antibiotic\tantibiotic target alteration\tErm(34)
""")


@pytest.fixture()
def minimal_dataset(tmp_path: Path) -> list[CARDRecord]:
    """Write minimal FASTA + TSV to tmp_path and return loaded records."""
    fasta_file = tmp_path / "test.fasta"
    tsv_file = tmp_path / "test_aro_index.tsv"
    fasta_file.write_text(MINIMAL_FASTA)
    tsv_file.write_text(MINIMAL_ARO_TSV)
    return load_card_dataset(fasta_file, tsv_file)


# ---------------------------------------------------------------------------
# Unit tests: FASTA header parsing
# ---------------------------------------------------------------------------


class TestParseFastaHeader:
    def test_standard_header(self):
        header = "gb|ACT97415.1|ARO:3002999|CblA-1 [mixed culture bacterium AX_gF3SD01_15]"
        result = _parse_fasta_header(header)
        assert result["protein_accession"] == "ACT97415.1"
        assert result["aro_accession"] == "ARO:3002999"
        assert result["gene_name"] == "CblA-1"
        assert result["organism"] == "mixed culture bacterium AX_gF3SD01_15"

    def test_gene_name_with_parentheses(self):
        header = "gb|AAC44793.1|ARO:3002524|AAC(2')-Ib [Providencia stuartii]"
        result = _parse_fasta_header(header)
        assert result["gene_name"] == "AAC(2')-Ib"
        assert result["organism"] == "Providencia stuartii"

    def test_malformed_header_raises(self):
        with pytest.raises(ValueError, match="Unexpected CARD FASTA header"):
            _parse_fasta_header("not_a_valid_header")


# ---------------------------------------------------------------------------
# Unit tests: minimal fixture dataset
# ---------------------------------------------------------------------------


class TestMinimalDataset:
    def test_record_count(self, minimal_dataset):
        assert len(minimal_dataset) == 3

    def test_record_type(self, minimal_dataset):
        assert all(isinstance(r, CARDRecord) for r in minimal_dataset)

    def test_aro_accessions_are_strings(self, minimal_dataset):
        for r in minimal_dataset:
            assert isinstance(r.aro_accession, str)
            assert r.aro_accession.startswith("ARO:")

    def test_sequences_are_non_empty_strings(self, minimal_dataset):
        for r in minimal_dataset:
            assert isinstance(r.sequence, str)
            assert len(r.sequence) > 0
            # Amino acid sequences should not contain nucleotide-only characters
            assert "T" not in r.sequence or any(
                aa in r.sequence for aa in "ACDEFGHIKLMNPQRSVWY"
            )

    def test_drug_classes_are_lists(self, minimal_dataset):
        for r in minimal_dataset:
            assert isinstance(r.drug_classes, list)
            assert all(isinstance(d, str) for d in r.drug_classes)

    def test_multi_drug_class_splitting(self, minimal_dataset):
        # Erm(34) has three semicolon-delimited drug classes in the fixture
        erm34 = next(r for r in minimal_dataset if r.gene_name == "Erm(34)")
        assert len(erm34.drug_classes) == 3
        assert "macrolide antibiotic" in erm34.drug_classes
        assert "lincosamide antibiotic" in erm34.drug_classes
        assert "streptogramin antibiotic" in erm34.drug_classes

    def test_resistance_mechanism_non_empty(self, minimal_dataset):
        for r in minimal_dataset:
            assert isinstance(r.resistance_mechanism, str)
            assert len(r.resistance_mechanism) > 0

    def test_metadata_aligned_to_aro(self, minimal_dataset):
        """Verify that metadata fields join correctly from the ARO index."""
        cbl = next(r for r in minimal_dataset if r.aro_accession == "ARO:3002999")
        assert cbl.gene_name == "CblA-1"
        assert cbl.resistance_mechanism == "antibiotic inactivation"
        assert cbl.amr_gene_family == "CblA beta-lactamase"


# ---------------------------------------------------------------------------
# Unit tests: label vocabulary builder
# ---------------------------------------------------------------------------


class TestGetLabelVocabularies:
    def test_keys_present(self, minimal_dataset):
        vocabs = get_label_vocabularies(minimal_dataset)
        assert "drug_class" in vocabs
        assert "resistance_mechanism" in vocabs
        assert "amr_gene_family" in vocabs

    def test_vocabularies_are_sorted(self, minimal_dataset):
        vocabs = get_label_vocabularies(minimal_dataset)
        for key, vocab in vocabs.items():
            assert vocab == sorted(vocab), f"Vocabulary '{key}' is not sorted"

    def test_no_duplicates(self, minimal_dataset):
        vocabs = get_label_vocabularies(minimal_dataset)
        for key, vocab in vocabs.items():
            assert len(vocab) == len(set(vocab)), f"Duplicates in '{key}' vocabulary"

    def test_multi_drug_class_expanded(self, minimal_dataset):
        vocabs = get_label_vocabularies(minimal_dataset)
        # All three drug classes from Erm(34) should be in the vocabulary
        assert "macrolide antibiotic" in vocabs["drug_class"]
        assert "lincosamide antibiotic" in vocabs["drug_class"]
        assert "streptogramin antibiotic" in vocabs["drug_class"]


# ---------------------------------------------------------------------------
# Integration tests: full CARD dataset (skipped when data absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_dataset():
    if not _REAL_DATA_AVAILABLE:
        pytest.skip("CARD data files not present (expected on CPU/GPU server)")
    return load_card_dataset(_FASTA_PATH, _ARO_INDEX_PATH)


@_skip_no_data
class TestFullCARDDataset:
    def test_record_count_matches_fasta(self, full_dataset):
        # CARD broadstreet v4.0.1 homolog model FASTA has 6052 sequences.
        assert len(full_dataset) == 6052

    def test_no_empty_sequences(self, full_dataset):
        assert all(len(r.sequence) > 0 for r in full_dataset)

    def test_no_empty_aro_accessions(self, full_dataset):
        assert all(r.aro_accession.startswith("ARO:") for r in full_dataset)

    def test_all_have_resistance_mechanism(self, full_dataset):
        missing = [r.aro_accession for r in full_dataset if not r.resistance_mechanism]
        assert missing == [], f"Records missing resistance_mechanism: {missing[:5]}"

    def test_all_have_at_least_one_drug_class(self, full_dataset):
        missing = [r.aro_accession for r in full_dataset if not r.drug_classes]
        assert missing == [], f"Records missing drug_class: {missing[:5]}"

    def test_aro_accessions_are_unique(self, full_dataset):
        accessions = [r.aro_accession for r in full_dataset]
        assert len(accessions) == len(set(accessions)), "Duplicate ARO accessions found"

    def test_label_vocabulary_sizes(self, full_dataset):
        vocabs = get_label_vocabularies(full_dataset)
        # Rough sanity bounds based on CARD v4.0.1 content
        assert len(vocabs["resistance_mechanism"]) >= 5
        assert len(vocabs["drug_class"]) >= 20
        assert len(vocabs["amr_gene_family"]) >= 100
