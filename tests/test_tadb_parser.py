"""Smoke tests for src/data/tadb_parser.py."""

from src.data.tadb_parser import parse_tadb_fasta

_DUMMY_SEQ = "MKTAYIAKQRQISFVKSHFSRQ"

_FASTA_CONTENT = f"""\
>T1 WP_000001.1 NC_000001:100-200 [Escherichia coli]
{_DUMMY_SEQ}
>AT1 WP_000002.1 NC_000001:c500-400 [Escherichia coli]
{_DUMMY_SEQ}
>T2 WP_000003.1 NC_000002:300-400
{_DUMMY_SEQ}
>T3 WP_000004.1 NC_000003 [Klebsiella pneumoniae]
{_DUMMY_SEQ}
"""


def test_parses_valid_headers_and_skips_malformed(tmp_path, caplog):
    fasta_path = tmp_path / "type_II_test.fas"
    fasta_path.write_text(_FASTA_CONTENT)

    loci = parse_tadb_fasta(fasta_path, source="exp")

    # T2 (missing organism brackets) and T3 (missing replicon:coords) are
    # malformed and should be warned-and-skipped, not raised.
    assert len(loci) == 2
    assert any("Skipping malformed" in r.message for r in caplog.records)


def test_toxin_plus_strand_fields(tmp_path):
    fasta_path = tmp_path / "type_II_test.fas"
    fasta_path.write_text(_FASTA_CONTENT)

    loci = parse_tadb_fasta(fasta_path, source="exp")
    t1 = next(l for l in loci if l.locus_id == "T1")

    assert t1.locus_type == "toxin"
    assert t1.protein_accession == "WP_000001.1"
    assert t1.replicon_accession == "NC_000001"
    assert t1.start == 100
    assert t1.end == 200
    assert t1.strand == "plus"
    assert t1.organism == "Escherichia coli"
    assert t1.source == "exp"


def test_antitoxin_minus_strand_coords_normalized_to_min_max(tmp_path):
    fasta_path = tmp_path / "type_II_test.fas"
    fasta_path.write_text(_FASTA_CONTENT)

    loci = parse_tadb_fasta(fasta_path, source="exp")
    at1 = next(l for l in loci if l.locus_id == "AT1")

    assert at1.locus_type == "antitoxin"
    # Header says "c500-400" (minus strand); start/end must be min/max, not
    # the raw (possibly descending) order the header gives them in.
    assert at1.start == 400
    assert at1.end == 500
    assert at1.strand == "minus"


def test_source_label_is_recorded(tmp_path):
    fasta_path = tmp_path / "type_II_test.fas"
    fasta_path.write_text(_FASTA_CONTENT)

    loci = parse_tadb_fasta(fasta_path, source="pre")
    assert all(l.source == "pre" for l in loci)
