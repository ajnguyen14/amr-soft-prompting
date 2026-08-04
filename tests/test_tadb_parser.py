"""Smoke tests for the TADB 3.0 Type II parser (tadb_parser.py)."""

import textwrap
from pathlib import Path

import pytest

from src.data.tadb_parser import (
    TADBLocus,
    load_all_tadb_loci,
    parse_tadb_fasta,
    replicon_accessions,
)

# ---------------------------------------------------------------------------
# Paths to the real TADB data -- tests requiring these are skipped if absent.
# ---------------------------------------------------------------------------
_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
_SOURCE_FILENAMES = (
    "type_II_T_exp.fas",
    "type_II_AT_exp.fas",
    "type_II_T_pre.fas",
    "type_II_AT_pre.fas",
)

_REAL_DATA_AVAILABLE = all((_RAW_DIR / f).exists() for f in _SOURCE_FILENAMES)
_skip_no_data = pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason="TADB 3.0 data files not present (expected on CPU/GPU server)",
)


# ---------------------------------------------------------------------------
# Fixture: minimal in-memory FASTA covering plus strand, minus strand
# (the "c" prefix reverses coordinate order), and the scientific-notation
# coordinate artifact seen in type_II_AT_pre.fas.
# ---------------------------------------------------------------------------

MINIMAL_TOXIN_FASTA = textwrap.dedent("""\
    >T73 NP_214813.1 NC_000962:363476-363778 [Mycobacterium tuberculosis H37Rv]
    MIAPGDIAPRRDSEHELYVAVLSNALHRAADTGRVITCPFIPGRVPEDLLAMVVAVEQPNGTLLPELVQWLHVAALGAPL
    >T28 WP_000916169.1 NC_000915:c946611-946345 [Helicobacter pylori 26695]
    VLKLNLKKSFQKDFDKLLLNGFDDSVLNEVILTLRKKEPLDPQFQDHALKGKWKPFRECHIKPDVLLVYLVKDDELILLR
    >T999 WP_999999999.1 NZ_CP094865:2e+06-2000257 [Staphylococcus epidermidis]
    MFGSKLSKTIAAFAALVSSAATMA
""")

MINIMAL_ANTITOXIN_FASTA = textwrap.dedent("""\
    >AT73 NP_214812.1 NC_000962:363252-363479 [Mycobacterium tuberculosis H37Rv]
    MTKEKISVTVDAAVLAAIDADARAAGLNRSEMIEQALRNEHLRVALRDYTAKTVPALDIDAYAQRVYQANRAAGS
""")

MALFORMED_FASTA = textwrap.dedent("""\
    >T1 WP_000916169.1 not_a_valid_coordinate_field
    MFGSKLSKTIAAFAALVSSAATMA
""")


@pytest.fixture()
def minimal_toxin_loci(tmp_path: Path) -> list[TADBLocus]:
    fasta_file = tmp_path / "toxin.fas"
    fasta_file.write_text(MINIMAL_TOXIN_FASTA)
    return parse_tadb_fasta(fasta_file, locus_type="toxin", confidence="exp")


# ---------------------------------------------------------------------------
# Unit tests: header parsing
# ---------------------------------------------------------------------------


class TestParseTadbFasta:
    def test_record_count(self, minimal_toxin_loci):
        assert len(minimal_toxin_loci) == 3

    def test_record_type_and_fields(self, minimal_toxin_loci):
        assert all(isinstance(locus, TADBLocus) for locus in minimal_toxin_loci)
        assert all(locus.locus_type == "toxin" for locus in minimal_toxin_loci)
        assert all(locus.confidence == "exp" for locus in minimal_toxin_loci)

    def test_plus_strand_coordinates(self, minimal_toxin_loci):
        locus = next(l for l in minimal_toxin_loci if l.locus_id == "T73")
        assert locus.strand == "+"
        assert locus.start == 363476
        assert locus.end == 363778
        assert locus.replicon_accession == "NC_000962"

    def test_minus_strand_coordinates_normalized(self, minimal_toxin_loci):
        # TADB lists minus-strand coordinates in reverse (larger-first) order --
        # start must still come out <= end after normalization.
        locus = next(l for l in minimal_toxin_loci if l.locus_id == "T28")
        assert locus.strand == "-"
        assert locus.start == 946345
        assert locus.end == 946611
        assert locus.start <= locus.end

    def test_scientific_notation_coordinate(self, minimal_toxin_loci):
        # Known TADB export artifact (type_II_AT_pre.fas row AT240719): one
        # coordinate written as "2e+06" instead of a plain integer.
        locus = next(l for l in minimal_toxin_loci if l.locus_id == "T999")
        assert locus.start == 2000000
        assert locus.end == 2000257

    def test_replicon_accession_unversioned(self, minimal_toxin_loci):
        for locus in minimal_toxin_loci:
            assert "." not in locus.replicon_accession

    def test_invalid_locus_type_raises(self, tmp_path):
        fasta_file = tmp_path / "toxin.fas"
        fasta_file.write_text(MINIMAL_TOXIN_FASTA)
        with pytest.raises(ValueError, match="locus_type"):
            parse_tadb_fasta(fasta_file, locus_type="not_a_type", confidence="exp")

    def test_malformed_header_skipped_not_raised(self, tmp_path):
        fasta_file = tmp_path / "malformed.fas"
        fasta_file.write_text(MALFORMED_FASTA)
        loci = parse_tadb_fasta(fasta_file, locus_type="toxin", confidence="exp")
        assert loci == []


# ---------------------------------------------------------------------------
# Unit tests: replicon_accessions helper
# ---------------------------------------------------------------------------


class TestReplicionAccessions:
    def test_returns_unique_set(self, minimal_toxin_loci):
        accs = replicon_accessions(minimal_toxin_loci)
        assert accs == {"NC_000962", "NC_000915", "NZ_CP094865"}


# ---------------------------------------------------------------------------
# Integration tests: full TADB 3.0 dataset (skipped when data absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_tadb_loci():
    if not _REAL_DATA_AVAILABLE:
        pytest.skip("TADB 3.0 data files not present (expected on CPU/GPU server)")
    return load_all_tadb_loci(_RAW_DIR)


@_skip_no_data
class TestFullTadbDataset:
    def test_expected_counts_per_confidence(self, full_tadb_loci):
        exp_loci = [l for l in full_tadb_loci if l.confidence == "exp"]
        pre_loci = [l for l in full_tadb_loci if l.confidence == "pre"]
        # 403 experimentally-validated toxin/antitoxin pairs (CLAUDE.md), 404
        # antitoxin entries (one extra AT with no paired T -- see TADB source data).
        assert len(exp_loci) == 403 + 404
        assert len(pre_loci) == 169035 + 169035

    def test_no_versioned_replicon_accessions(self, full_tadb_loci):
        versioned = [l for l in full_tadb_loci if "." in l.replicon_accession]
        assert versioned == [], f"Unexpected versioned accessions: {versioned[:5]}"

    def test_all_start_le_end(self, full_tadb_loci):
        bad = [l for l in full_tadb_loci if l.start > l.end]
        assert bad == [], f"start > end for {len(bad)} loci, e.g. {bad[:3]}"

    def test_strand_values(self, full_tadb_loci):
        assert all(l.strand in ("+", "-") for l in full_tadb_loci)

    def test_replicon_accessions_nonempty(self, full_tadb_loci):
        accs = replicon_accessions(full_tadb_loci)
        assert len(accs) > 0
