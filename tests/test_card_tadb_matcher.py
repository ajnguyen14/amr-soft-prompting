"""Smoke tests for the CARD/TADB accession matcher (card_tadb_matcher.py)."""

import textwrap
from pathlib import Path

import pytest

from src.data.card_tadb_matcher import (
    AccessionMatch,
    match_card_to_tadb_replicons,
    strip_accession_version,
)
from src.data.tadb_parser import TADBLocus, load_all_tadb_loci

# ---------------------------------------------------------------------------
# Paths to the real CARD + TADB data -- integration test skipped if absent.
# ---------------------------------------------------------------------------
_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
_ARO_INDEX_PATH = _RAW_DIR / "aro_index.tsv"
_TADB_FILENAMES = (
    "type_II_T_exp.fas",
    "type_II_AT_exp.fas",
    "type_II_T_pre.fas",
    "type_II_AT_pre.fas",
)
_REAL_DATA_AVAILABLE = _ARO_INDEX_PATH.exists() and all(
    (_RAW_DIR / f).exists() for f in _TADB_FILENAMES
)
_skip_no_data = pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason="CARD/TADB data files not present (expected on CPU/GPU server)",
)


def _make_locus(replicon_accession: str) -> TADBLocus:
    """Build a minimal TADBLocus with only replicon_accession under test."""
    return TADBLocus(
        locus_id="T1",
        locus_type="toxin",
        confidence="exp",
        protein_accession="WP_000000000.1",
        replicon_accession=replicon_accession,
        start=100,
        end=200,
        strand="+",
        organism="Test organism",
    )


# ---------------------------------------------------------------------------
# Fixture: minimal in-memory aro_index.tsv -- one row matches a fake TADB
# replicon (version-stripped), one has a DNA Accession absent from TADB, and
# one has no DNA Accession at all.
# ---------------------------------------------------------------------------

MINIMAL_ARO_TSV = textwrap.dedent("""\
    ARO Accession\tCVTERM ID\tModel Sequence ID\tModel ID\tModel Name\tARO Name\tProtein Accession\tDNA Accession\tAMR Gene Family\tDrug Class\tResistance Mechanism\tCARD Short Name
    ARO:3000248\t1\t1\t1\tOXA-1\tOXA-1\tAAA25892.1\tNC_000913.3\tOXA beta-lactamase\tcephalosporin\tantibiotic inactivation\tOXA-1
    ARO:3002999\t2\t2\t2\tCblA-1\tCblA-1\tACT97415.1\tGU256745.1\tCblA beta-lactamase\tcephalosporin\tantibiotic inactivation\tCblA-1
    ARO:3002524\t3\t3\t3\tAAC(2')-Ib\tAAC(2')-Ib\tAAC44793.1\t\tAAC(2')\taminoglycoside antibiotic\tantibiotic inactivation\tAAC(2')-Ib
""")


@pytest.fixture()
def minimal_matches(tmp_path: Path) -> list[AccessionMatch]:
    tsv_file = tmp_path / "test_aro_index.tsv"
    tsv_file.write_text(MINIMAL_ARO_TSV)
    # NC_000913 (unversioned) is the only replicon present in TADB here --
    # matches ARO:3000248's version-stripped NC_000913.3.
    fake_loci = [_make_locus("NC_000913")]
    return match_card_to_tadb_replicons(tsv_file, fake_loci)


# ---------------------------------------------------------------------------
# Unit tests: strip_accession_version
# ---------------------------------------------------------------------------


class TestStripAccessionVersion:
    def test_strips_version_suffix(self):
        assert strip_accession_version("AL123456.3") == "AL123456"

    def test_no_version_suffix_passthrough(self):
        assert strip_accession_version("NC_000913") == "NC_000913"

    def test_strips_only_last_suffix(self):
        # Accession names can legitimately contain dots elsewhere (rare, but
        # rsplit on the *last* dot is the correct behavior either way).
        assert strip_accession_version("AB.CD.3") == "AB.CD"


# ---------------------------------------------------------------------------
# Unit tests: match_card_to_tadb_replicons
# ---------------------------------------------------------------------------


class TestMatchCardToTadbReplicons:
    def test_only_matching_accession_returned(self, minimal_matches):
        assert len(minimal_matches) == 1
        assert minimal_matches[0].aro_accession == "ARO:3000248"

    def test_record_type(self, minimal_matches):
        assert all(isinstance(m, AccessionMatch) for m in minimal_matches)

    def test_versioned_accession_preserved_alongside_base(self, minimal_matches):
        match = minimal_matches[0]
        assert match.card_dna_accession == "NC_000913.3"
        assert match.base_accession == "NC_000913"

    def test_empty_dna_accession_excluded(self, tmp_path):
        tsv_file = tmp_path / "test_aro_index.tsv"
        tsv_file.write_text(MINIMAL_ARO_TSV)
        matches = match_card_to_tadb_replicons(tsv_file, [_make_locus("NC_000913")])
        matched_aros = {m.aro_accession for m in matches}
        assert "ARO:3002524" not in matched_aros  # empty DNA Accession

    def test_non_matching_accession_excluded(self, tmp_path):
        tsv_file = tmp_path / "test_aro_index.tsv"
        tsv_file.write_text(MINIMAL_ARO_TSV)
        matches = match_card_to_tadb_replicons(tsv_file, [_make_locus("NC_000913")])
        matched_aros = {m.aro_accession for m in matches}
        assert "ARO:3002999" not in matched_aros  # GU256745 not in TADB here

    def test_no_tadb_loci_yields_no_matches(self, tmp_path):
        tsv_file = tmp_path / "test_aro_index.tsv"
        tsv_file.write_text(MINIMAL_ARO_TSV)
        assert match_card_to_tadb_replicons(tsv_file, []) == []


# ---------------------------------------------------------------------------
# Integration test: full CARD + TADB datasets (skipped when data absent)
# ---------------------------------------------------------------------------


@_skip_no_data
class TestFullMatch:
    def test_expected_match_count(self):
        # Direct-accession prefilter coverage on CARD broadstreet v4.0.1 +
        # TADB 3.0 Type II (exp + pre): 145/6052 ARO accessions matched.
        loci = load_all_tadb_loci(_RAW_DIR)
        matches = match_card_to_tadb_replicons(_ARO_INDEX_PATH, loci)
        assert len(matches) == 145

    def test_matched_aro_accessions_are_unique(self):
        loci = load_all_tadb_loci(_RAW_DIR)
        matches = match_card_to_tadb_replicons(_ARO_INDEX_PATH, loci)
        aros = [m.aro_accession for m in matches]
        assert len(aros) == len(set(aros))

    def test_all_base_accessions_unversioned(self):
        loci = load_all_tadb_loci(_RAW_DIR)
        matches = match_card_to_tadb_replicons(_ARO_INDEX_PATH, loci)
        assert all("." not in m.base_accession for m in matches)
