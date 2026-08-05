"""Smoke tests for blast_runner.py.

Runs real (not mocked) BLAST+ binaries against tiny synthetic FASTA files --
correctness of tblastn's -outfmt 6 parsing is the thing most likely to be
subtly wrong, so a mock would risk hiding exactly the bug this needs to
catch. Skipped entirely if BLAST+ isn't installed (matches the project's
real-data skip pattern used elsewhere, e.g. test_card_tadb_matcher.py).
"""

import shutil
from pathlib import Path

import pytest

from src.data.blast_runner import (
    BlastHit,
    blast_card_against_representatives,
    build_blast_db,
    run_tblastn,
)

# ---------------------------------------------------------------------------
# BLAST+ availability -- these tests need real makeblastdb/tblastn binaries.
# ---------------------------------------------------------------------------
_BLAST_BIN_DIR = None
if shutil.which("makeblastdb") and shutil.which("tblastn"):
    _BLAST_BIN_DIR = None  # already on PATH
else:
    # Fall back to the known install location on spark-833c, if present.
    _candidate = Path.home() / "tools" / "ncbi-blast-2.17.0+" / "bin"
    if (_candidate / "makeblastdb").exists() and (_candidate / "tblastn").exists():
        _BLAST_BIN_DIR = _candidate

_BLAST_AVAILABLE = _BLAST_BIN_DIR is not None or bool(
    shutil.which("makeblastdb") and shutil.which("tblastn")
)
_skip_no_blast = pytest.mark.skipif(
    not _BLAST_AVAILABLE, reason="BLAST+ (makeblastdb/tblastn) not installed"
)

# ---------------------------------------------------------------------------
# Fixture data: a real CARD gene (CblA-1, ARO:3002999) embedded in a tiny
# synthetic "genome" with flanking junk on both sides, so the expected hit
# coordinates are known exactly (junk length + 1 through junk length + gene
# length, plus strand).
# ---------------------------------------------------------------------------
_GENE_DNA = (
    "ATGAAAGCATATTTCATCGCCATACTTACCTTATTCACTTGTATAGCTACCGTCGTCCGGGCGCAGCAAATGTCTGAACTTGAAAACCGGATTGACAGT"
    "CTGCTCAATGGCAAGAAAGCCACCGTTGGTATAGCCGTATGGACAGACAAAGGAGACATGCTCCGGTATAACGACCATGTACACTTCCCCTTGCTCAGT"
    "GTATTCAAATTCCATGTGGCACTGGCCGTACTGGACAAGATGGATAAGCAAAGCATCAGTCTGGACAGCATTGTTTCCATAAAGGCATCCCAAATGCCG"
    "CCCAATACCTACAGCCCCCTGCGGAAGAAGTTTCCCGACCAGGATTTCACGATTACGCTTAGGGAACTGATGCAATACAGCATTTCCCAAAGCGACAAC"
    "AATGCCTGCGACATCTTGATAGAATATGCAGGAGGCATCAAACATATCAACGACTATATCCACCGGTTGAGTATCGACTCCTTCAACCTCTCGGAAACA"
    "GAAGACGGCATGCACTCCAGCTTCGAGGCTGTATACCGCAACTGGAGTACTCCTTCCGCTATGGTCCGACTACTGAGAACGGCTGATGAAAAAGAGTTG"
    "TTCTCCAACAAGGAGCTGAAAGACTTCTTGTGGCAGACCATGATAGATACTGAAACCGGTGCCAACAAACTGAAAGGTATGTTGCCAGCCAAAACCGTG"
    "GTAGGACACAAGACCGGCTCTTCCGACCGCAATGCCGACGGTATGAAAACTGCAGATAATGATGCCGGCCTCGTTATCCTTCCCGACGGCCGGAAATAC"
    "TACATTGCCGCCTTCGTCATGGACTCATACGAGACGGATGAGGACAATGCGAACATCATCGCCCGCATATCACGCATGGTATATGATGCGATGAGATGA"
)
_GENE_PROTEIN = (
    "MKAYFIAILTLFTCIATVVRAQQMSELENRIDSLLNGKKATVGIAVWTDKGDMLRYNDHVHFPLLSVFKFHVALAVLDKMDKQSISLDSIVSIKASQMP"
    "PNTYSPLRKKFPDQDFTITLRELMQYSISQSDNNACDILIEYAGGIKHINDYIHRLSIDSFNLSETEDGMHSSFEAVYRNWSTPSAMVRLLRTADEKEL"
    "FSNKELKDFLWQTMIDTETGANKLKGMLPAKTVVGHKTGSSDRNADGMKTADNDAGLVILPDGRKYYIAAFVMDSYETDEDNANIIARISRMVYDAMR"
)
_UPSTREAM_JUNK = "GATTACA" * 20  # 140 bp of unrelated flanking sequence
_DOWNSTREAM_JUNK = "CATTAGG" * 20  # 140 bp of unrelated flanking sequence
_SYNTHETIC_GENOME = _UPSTREAM_JUNK + _GENE_DNA + _DOWNSTREAM_JUNK
_EXPECTED_START = len(_UPSTREAM_JUNK) + 1  # 1-based
# tblastn's protein query has no stop-codon symbol, so the aligned subject
# range stops 3bp short of the full CDS (the stop codon itself is never part
# of the alignment) -- confirmed empirically, not assumed.
_EXPECTED_END = len(_UPSTREAM_JUNK) + len(_GENE_DNA) - 3


@pytest.fixture()
def genome_fasta(tmp_path: Path) -> Path:
    path = tmp_path / "TEST000001.1.fasta"
    path.write_text(f">TEST000001.1 synthetic test replicon\n{_SYNTHETIC_GENOME}\n")
    return path


@pytest.fixture()
def query_fasta(tmp_path: Path) -> Path:
    path = tmp_path / "query.fasta"
    path.write_text(f">ARO:3002999\n{_GENE_PROTEIN}\n")
    return path


@pytest.fixture()
def blast_db(tmp_path: Path, genome_fasta: Path) -> Path:
    db_path = tmp_path / "blastdb" / "TEST000001.1"
    build_blast_db(genome_fasta, db_path, blast_bin_dir=_BLAST_BIN_DIR)
    return db_path


# ---------------------------------------------------------------------------
# Unit tests: build_blast_db + run_tblastn against a known-answer synthetic genome
# ---------------------------------------------------------------------------


@_skip_no_blast
class TestRunTblastn:
    def test_finds_expected_hit(self, query_fasta, blast_db):
        hits = run_tblastn(query_fasta, blast_db, blast_bin_dir=_BLAST_BIN_DIR)
        assert len(hits) == 1
        assert isinstance(hits[0], BlastHit)

    def test_hit_coordinates_match_known_insertion_point(self, query_fasta, blast_db):
        hit = run_tblastn(query_fasta, blast_db, blast_bin_dir=_BLAST_BIN_DIR)[0]
        assert hit.start == _EXPECTED_START
        assert hit.end == _EXPECTED_END
        assert hit.strand == "plus"

    def test_hit_identity_and_coverage_near_perfect(self, query_fasta, blast_db):
        hit = run_tblastn(query_fasta, blast_db, blast_bin_dir=_BLAST_BIN_DIR)[0]
        assert hit.percent_identity > 99.0
        assert hit.query_coverage > 99.0

    def test_hit_metadata_fields(self, query_fasta, blast_db):
        hit = run_tblastn(query_fasta, blast_db, blast_bin_dir=_BLAST_BIN_DIR)[0]
        assert hit.aro_accession == "ARO:3002999"
        assert hit.replicon_accession == "TEST000001.1"

    def test_below_identity_threshold_excluded(self, query_fasta, blast_db):
        hits = run_tblastn(
            query_fasta, blast_db, blast_bin_dir=_BLAST_BIN_DIR, min_identity=101.0
        )
        assert hits == []

    def test_unrelated_query_produces_no_hit(self, tmp_path, blast_db):
        unrelated = tmp_path / "unrelated.fasta"
        unrelated.write_text(">ARO:9999999\n" + "W" * 50 + "\n")
        hits = run_tblastn(unrelated, blast_db, blast_bin_dir=_BLAST_BIN_DIR)
        assert hits == []


# ---------------------------------------------------------------------------
# Unit tests: blast_card_against_representatives orchestration
# ---------------------------------------------------------------------------


@_skip_no_blast
class TestBlastCardAgainstRepresentatives:
    def test_end_to_end_grouped_run(self, tmp_path, genome_fasta):
        refseq_dir = tmp_path / "refseq"
        refseq_dir.mkdir()
        shutil.copy(genome_fasta, refseq_dir / "TEST000001.1.fasta")

        blastdb_dir = tmp_path / "blastdb"
        query_sequences_by_group = {"TEST000001.1": {"ARO:3002999": _GENE_PROTEIN}}

        hits = blast_card_against_representatives(
            query_sequences_by_group,
            refseq_dir=refseq_dir,
            blastdb_dir=blastdb_dir,
            blast_bin_dir=_BLAST_BIN_DIR,
        )

        assert len(hits) == 1
        assert hits[0].aro_accession == "ARO:3002999"
        assert hits[0].start == _EXPECTED_START
        assert hits[0].end == _EXPECTED_END

    def test_missing_fetched_fasta_skipped_not_raised(self, tmp_path):
        refseq_dir = tmp_path / "refseq"
        refseq_dir.mkdir()  # empty -- no FASTA files fetched
        blastdb_dir = tmp_path / "blastdb"
        query_sequences_by_group = {"MISSING000001.1": {"ARO:0000001": _GENE_PROTEIN}}

        hits = blast_card_against_representatives(
            query_sequences_by_group,
            refseq_dir=refseq_dir,
            blastdb_dir=blastdb_dir,
            blast_bin_dir=_BLAST_BIN_DIR,
        )

        assert hits == []
