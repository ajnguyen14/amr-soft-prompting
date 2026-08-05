"""BLAST CARD protein sequences against fetched RefSeq representative genomes
to place each ARO accession at a genomic coordinate (CLAUDE.md TA-Proximity
Pipeline Step 1, second half).

Queries are CARD amino acid sequences; subjects are the nucleotide FASTA
records fetched by refseq_fetch.py, so this uses `tblastn` (protein query vs.
translated nucleotide subject), not `blastn`. Wraps the BLAST+ command-line
tools via subprocess -- there is no first-class BLAST+ Python binding that
ships with biopython (Bio.Blast only parses output, it doesn't run BLAST).

BLASTing is batched per representative accession, not per CARD protein: all
ARO entries mapped to the same representative (see
refseq_representative.map_aro_to_representative) are BLASTed in a single
tblastn call against a single BLAST database built for that representative,
so this makes ~738 subprocess calls total instead of 6404.
"""

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# tblastn tabular output columns, in the order requested via -outfmt below.
# sstrand and qcovs are BLAST+-computed convenience fields (not raw HSP data)
# that avoid recomputing strand/coverage by hand from raw coordinates.
_OUTFMT_COLUMNS = (
    "qseqid",
    "sseqid",
    "pident",
    "sstart",
    "send",
    "sstrand",
    "evalue",
    "bitscore",
    "qcovs",
)
_OUTFMT = "6 " + " ".join(_OUTFMT_COLUMNS)


@dataclass(frozen=True)
class BlastHit:
    """One best-hit BLAST placement of a CARD protein on a replicon.

    Args:
        aro_accession: CARD ARO accession (e.g. 'ARO:3002999') -- the BLAST
            query ID, since query FASTA headers are written as ARO accessions.
        replicon_accession: Subject accession the hit landed on (the
            representative accession's own, e.g. 'AA000001.1').
        start: 1-based start coordinate on the replicon (always <= end,
            strand tracked separately -- same normalization as
            tadb_parser.TADBLocus, so Step 3's same-replicon comparison
            doesn't need special-casing per source).
        end: 1-based end coordinate on the replicon.
        strand: 'plus' or 'minus'.
        percent_identity: BLAST pident for the hit, in [0, 100].
        evalue: BLAST e-value for the hit.
        query_coverage: BLAST qcovs for the hit, in [0, 100].
    """

    aro_accession: str
    replicon_accession: str
    start: int
    end: int
    strand: str
    percent_identity: float
    evalue: float
    query_coverage: float


def build_blast_db(fasta_path: str | Path, db_path: str | Path, blast_bin_dir: Optional[str | Path] = None) -> None:
    """Build a nucleotide BLAST database from a fetched RefSeq FASTA file.

    Args:
        fasta_path: Path to a single-record nucleotide FASTA (e.g.
            data/raw/refseq/<accession>.fasta from refseq_fetch.py).
        db_path: Output path prefix for the BLAST database files.
        blast_bin_dir: Directory containing the makeblastdb executable. If
            None, relies on makeblastdb being on PATH.

    Raises:
        subprocess.CalledProcessError: If makeblastdb exits non-zero.
    """
    makeblastdb = str(Path(blast_bin_dir) / "makeblastdb") if blast_bin_dir else "makeblastdb"
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [makeblastdb, "-in", str(fasta_path), "-dbtype", "nucl", "-out", str(db_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def _parse_tblastn_line(line: str) -> BlastHit:
    """Parse one tblastn -outfmt 6 line into a BlastHit.

    Normalizes sstart/send to start <= end with strand tracked separately --
    tblastn reports sstart > send for minus-strand hits, so a plain
    coordinate pair without normalization would be ambiguous to compare
    against TADB's already-normalized loci.

    Args:
        line: One tab-delimited line of tblastn output, columns per
            _OUTFMT_COLUMNS.

    Returns:
        The parsed BlastHit.
    """
    fields = line.rstrip("\n").split("\t")
    qseqid, sseqid, pident, sstart, send, sstrand, evalue, bitscore, qcovs = fields

    start, end = int(sstart), int(send)
    if start > end:
        start, end = end, start

    return BlastHit(
        aro_accession=qseqid,
        replicon_accession=sseqid,
        start=start,
        end=end,
        strand=sstrand,
        percent_identity=float(pident),
        evalue=float(evalue),
        query_coverage=float(qcovs),
    )


def run_tblastn(
    query_fasta_path: str | Path,
    db_path: str | Path,
    blast_bin_dir: Optional[str | Path] = None,
    min_identity: float = 95.0,
    min_query_coverage: float = 90.0,
) -> list[BlastHit]:
    """Run tblastn and return one best hit per query that clears both thresholds.

    A query can produce multiple HSPs (e.g. repeated domains); this keeps
    only the lowest-e-value hit per query ID among those passing the
    identity/coverage thresholds, so callers get at most one BlastHit per
    query sequence.

    Args:
        query_fasta_path: Path to a (possibly multi-record) protein FASTA.
        db_path: BLAST database path prefix, e.g. from build_blast_db.
        blast_bin_dir: Directory containing the tblastn executable. If None,
            relies on tblastn being on PATH.
        min_identity: Minimum percent identity to accept a hit (CLAUDE.md
            TA-Proximity Pipeline Step 1: conservative default, not yet
            finalized -- 95% here).
        min_query_coverage: Minimum percent query coverage to accept a hit.

    Returns:
        List of best BlastHit per query ID that passed both thresholds.
        Queries with no passing hit are simply absent from the result --
        this is Step 1's real BLAST-mapping failure case ('unknown' in the
        eventual TA-proximity vocabulary), distinct from a query that maps
        cleanly but has no nearby TA locus ('no_ta_locus').

    Raises:
        subprocess.CalledProcessError: If tblastn exits non-zero.
    """
    tblastn = str(Path(blast_bin_dir) / "tblastn") if blast_bin_dir else "tblastn"

    proc = subprocess.run(
        [
            tblastn,
            "-query", str(query_fasta_path),
            "-db", str(db_path),
            "-outfmt", _OUTFMT,
            "-evalue", "1e-10",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # BLAST's tabular output is already sorted by e-value ascending within
    # each query in practice, but that ordering isn't a documented guarantee,
    # so the best hit per query is picked explicitly rather than relied on.
    best_hit_by_query: dict[str, BlastHit] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        hit = _parse_tblastn_line(line)
        if hit.percent_identity < min_identity or hit.query_coverage < min_query_coverage:
            continue
        current_best = best_hit_by_query.get(hit.aro_accession)
        if current_best is None or hit.evalue < current_best.evalue:
            best_hit_by_query[hit.aro_accession] = hit

    return list(best_hit_by_query.values())


def blast_card_against_representatives(
    query_sequences_by_group: dict[str, dict[str, str]],
    refseq_dir: str | Path,
    blastdb_dir: str | Path,
    blast_bin_dir: Optional[str | Path] = None,
    min_identity: float = 95.0,
    min_query_coverage: float = 90.0,
) -> list[BlastHit]:
    """BLAST CARD proteins against their organism group's representative genome.

    One BLAST database build + one tblastn call per representative accession
    (not per CARD protein), batching all of that group's queries into a
    single multi-FASTA per call.

    Args:
        query_sequences_by_group: Dict mapping representative accession to a
            dict of {aro_accession: protein_sequence} for every ARO entry in
            that organism group (see
            refseq_representative.map_aro_to_representative for how to build
            this grouping, and card_parser.load_card_dataset for sequences).
        refseq_dir: Directory containing one <accession>.fasta per
            representative accession (from refseq_fetch.py).
        blastdb_dir: Directory to write BLAST database files into.
        blast_bin_dir: Directory containing makeblastdb/tblastn executables.
            If None, relies on both being on PATH.
        min_identity: Passed through to run_tblastn.
        min_query_coverage: Passed through to run_tblastn.

    Returns:
        All BlastHit results across every representative group, one entry
        per ARO accession that BLAST-mapped successfully.
    """
    all_hits: list[BlastHit] = []
    refseq_dir = Path(refseq_dir)
    blastdb_dir = Path(blastdb_dir)

    for representative_accession, sequences_by_aro in query_sequences_by_group.items():
        fasta_path = refseq_dir / f"{representative_accession}.fasta"
        if not fasta_path.exists():
            logger.warning(
                "No fetched FASTA for representative accession %s -- skipping %d ARO entries",
                representative_accession,
                len(sequences_by_aro),
            )
            continue

        db_path = blastdb_dir / representative_accession
        build_blast_db(fasta_path, db_path, blast_bin_dir=blast_bin_dir)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as query_file:
            for aro_accession, sequence in sequences_by_aro.items():
                query_file.write(f">{aro_accession}\n{sequence}\n")
            query_fasta_path = query_file.name

        try:
            hits = run_tblastn(
                query_fasta_path,
                db_path,
                blast_bin_dir=blast_bin_dir,
                min_identity=min_identity,
                min_query_coverage=min_query_coverage,
            )
        finally:
            Path(query_fasta_path).unlink()

        all_hits.extend(hits)

    total_queries = sum(len(v) for v in query_sequences_by_group.values())
    logger.info(
        "BLAST-mapped %d/%d CARD ARO accessions (%.1f%% coverage) across %d representative groups",
        len(all_hits),
        total_queries,
        (len(all_hits) / total_queries * 100) if total_queries else 0.0,
        len(query_sequences_by_group),
    )
    return all_hits
