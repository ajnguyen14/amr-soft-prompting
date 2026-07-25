"""Smoke tests for src/data/blast_runner.py.

Builds a tiny synthetic BLAST database (one short fake "genome") rather than
using the real 293-genome RefSeq database, so this stays fast and has no
external dependencies beyond the BLAST+ binaries themselves.
"""

import random

import pytest
from Bio.Seq import Seq

from src.data.blast_runner import build_blast_database, map_card_to_refseq
from src.data.card_parser import CARDRecord

# Non-stop codons only, so a random codon string always translates cleanly.
_CODONS = [
    c for c in (a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT")
    if c not in ("TAA", "TAG", "TGA")
]


def _random_orf(num_codons: int, seed: int) -> tuple[str, str]:
    """Return (nucleotide_sequence, translated_protein) for a random ORF."""
    rng = random.Random(seed)
    nt = "".join(rng.choice(_CODONS) for _ in range(num_codons))
    protein = str(Seq(nt).translate())
    return nt, protein


def _make_card_record(aro_accession: str, sequence: str) -> CARDRecord:
    """Build a CARDRecord with placeholder metadata blast_runner doesn't read."""
    return CARDRecord(
        aro_accession=aro_accession,
        protein_accession="TEST.1",
        gene_name="testGene",
        organism="Test organism",
        sequence=sequence,
        drug_classes=["test drug class"],
        resistance_mechanism="test mechanism",
        amr_gene_family="test family",
        card_short_name="testGene",
    )


@pytest.fixture
def fake_genome_db(tmp_path):
    """A tiny 1-replicon BLAST database with one known ORF embedded in filler."""
    orf_nt, orf_protein = _random_orf(num_codons=40, seed=42)
    flank_nt, _ = _random_orf(num_codons=100, seed=1)
    genome_seq = flank_nt[:150] + orf_nt + flank_nt[150:]

    genomes_dir = tmp_path / "genomes"
    genomes_dir.mkdir()
    (genomes_dir / "FAKE_001.fna").write_text(f">FAKE_CHR1 synthetic test replicon\n{genome_seq}\n")

    db_path = build_blast_database(genomes_dir, tmp_path / "blastdb" / "test_db")
    expected_start = 150 + 1  # 1-based
    expected_end = 150 + len(orf_nt)
    return db_path, orf_protein, expected_start, expected_end


def test_matching_protein_maps_to_expected_coordinates(fake_genome_db):
    db_path, orf_protein, expected_start, expected_end = fake_genome_db
    records = [_make_card_record("ARO:TEST1", orf_protein)]

    results, coverage_pct = map_card_to_refseq(
        records, db_path, min_pident=95.0, min_qcov=90.0, max_evalue=1e-10,
    )

    result = results["ARO:TEST1"]
    assert result.mapped is True
    assert result.replicon_accession == "FAKE_CHR1"
    assert result.start == expected_start
    assert result.end == expected_end
    assert result.strand == "plus"
    assert result.pident == pytest.approx(100.0)
    assert result.qcov == pytest.approx(100.0)
    assert coverage_pct == pytest.approx(100.0)


def test_unrelated_protein_is_unmapped(fake_genome_db):
    db_path, _, _, _ = fake_genome_db
    _, unrelated_protein = _random_orf(num_codons=40, seed=999)
    records = [_make_card_record("ARO:TEST2", unrelated_protein)]

    results, coverage_pct = map_card_to_refseq(
        records, db_path, min_pident=95.0, min_qcov=90.0, max_evalue=1e-10,
    )

    result = results["ARO:TEST2"]
    assert result.mapped is False
    assert result.replicon_accession is None
    assert result.start is None
    assert result.pident is None
    assert coverage_pct == pytest.approx(0.0)


def test_mixed_batch_reports_partial_coverage(fake_genome_db):
    db_path, orf_protein, _, _ = fake_genome_db
    _, unrelated_protein = _random_orf(num_codons=40, seed=999)
    records = [
        _make_card_record("ARO:TEST1", orf_protein),
        _make_card_record("ARO:TEST2", unrelated_protein),
    ]

    results, coverage_pct = map_card_to_refseq(
        records, db_path, min_pident=95.0, min_qcov=90.0, max_evalue=1e-10,
    )

    assert results["ARO:TEST1"].mapped is True
    assert results["ARO:TEST2"].mapped is False
    assert coverage_pct == pytest.approx(50.0)
