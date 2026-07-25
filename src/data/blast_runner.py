"""BLAST-map CARD protein sequences onto RefSeq genomic coordinates.

Step 1 of the V2 TA-proximity pipeline (see CLAUDE.md's "TA-Proximity
Pipeline" section): places each CARD ARO accession at a genomic coordinate
(replicon_accession:start-end) by searching its protein sequence against a
BLAST database built from `scripts/fetch_refseq_genomes.py`'s downloaded
genomes. Downstream, `tadb_parser.py`/`ta_proximity.py` join these
coordinates against TADB 3.0 loci to compute same-replicon bp distance.

CARD's sequences are amino acid (protein) and the RefSeq genomes are
nucleotide, so the search direction is protein-query-vs-nucleotide-subject:
`tblastn`, not `blastp`.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.data.card_parser import CARDRecord

# tblastn tabular output columns requested via -outfmt. qcovs (query coverage
# per subject) requires no extra flag beyond naming it here.
_OUTFMT_FIELDS = [
    "qseqid", "sseqid", "pident", "qcovs", "evalue", "bitscore", "sstart", "send", "sstrand",
]


@dataclass
class BlastMapping:
    """Result of mapping one CARD ARO accession onto the RefSeq BLAST database.

    Args:
        aro_accession: ARO ontology accession (e.g. 'ARO:3002999').
        mapped: Whether a hit passed the identity/coverage/e-value thresholds.
        replicon_accession: RefSeq nucleotide accession of the best hit (e.g.
            'NC_000913.3'), or None if unmapped.
        start: Lower genomic coordinate of the hit on that replicon, or None.
        end: Higher genomic coordinate of the hit on that replicon, or None.
        strand: 'plus' or 'minus', or None if unmapped.
        pident: Percent identity of the best hit, or None if unmapped.
        qcov: Percent query coverage of the best hit, or None if unmapped.
        evalue: E-value of the best hit, or None if unmapped.
    """

    aro_accession: str
    mapped: bool
    replicon_accession: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    strand: Optional[str] = None
    pident: Optional[float] = None
    qcov: Optional[float] = None
    evalue: Optional[float] = None


def build_blast_database(genomes_dir: str | Path, db_output_prefix: str | Path) -> Path:
    """Concatenate downloaded RefSeq genomes and build a nucleotide BLAST database.

    Args:
        genomes_dir: Directory of per-accession `.fna` files, as written by
            scripts/fetch_refseq_genomes.py.
        db_output_prefix: Path prefix for the BLAST database files (passed to
            makeblastdb's -out); the parent directory is created if needed.

    Returns:
        The same db_output_prefix, as a Path, for use as `run_tblastn`'s db_path.

    Raises:
        subprocess.CalledProcessError: If makeblastdb fails.
    """
    genomes_dir = Path(genomes_dir)
    db_output_prefix = Path(db_output_prefix)
    db_output_prefix.parent.mkdir(parents=True, exist_ok=True)

    combined_fasta = db_output_prefix.parent / "combined_refseq.fna"
    with open(combined_fasta, "wb") as out:
        for fna_path in sorted(genomes_dir.glob("*.fna")):
            out.write(fna_path.read_bytes())

    subprocess.run(
        [
            "makeblastdb", "-in", str(combined_fasta), "-dbtype", "nucl",
            "-parse_seqids", "-out", str(db_output_prefix),
        ],
        check=True, capture_output=True, text=True,
    )
    return db_output_prefix


def _write_query_fasta(records: list[CARDRecord], out_path: Path) -> None:
    """Write CARD records to a FASTA file keyed by ARO accession.

    Using aro_accession (rather than CARD's compound `gb|...|ARO:...|...`
    header) as the sequence ID means tblastn's qseqid column is directly the
    dict key `map_card_to_refseq` groups hits by -- no header re-parsing needed.
    """
    with open(out_path, "w") as f:
        for record in records:
            f.write(f">{record.aro_accession}\n{record.sequence}\n")


def run_tblastn(
    query_fasta_path: str | Path,
    db_path: str | Path,
    max_evalue: float,
    max_target_seqs: int = 5,
    num_threads: int = 1,
) -> list[dict[str, str]]:
    """Run tblastn and parse its tabular output.

    Args:
        query_fasta_path: Path to a protein FASTA (queries).
        db_path: Prefix of a nucleotide BLAST database (from build_blast_database).
        max_evalue: tblastn's native -evalue cutoff. Coarser than the
            pident/qcov filtering map_card_to_refseq applies afterward, since
            tblastn has no built-in identity/coverage threshold flags.
        max_target_seqs: Max hits to keep per query.
        num_threads: tblastn's -num_threads. Parallelizes across queries (this
            workload is many independent queries against one database, BLAST+'s
            best case for threading), not by splitting a single query.

    Returns:
        One dict per hit row, with keys matching _OUTFMT_FIELDS (all values as
        raw strings; map_card_to_refseq handles numeric parsing).

    Raises:
        subprocess.CalledProcessError: If tblastn fails.
    """
    result = subprocess.run(
        [
            "tblastn", "-query", str(query_fasta_path), "-db", str(db_path),
            "-outfmt", "6 " + " ".join(_OUTFMT_FIELDS),
            "-evalue", str(max_evalue), "-max_target_seqs", str(max_target_seqs),
            "-num_threads", str(num_threads),
        ],
        check=True, capture_output=True, text=True,
    )
    rows = []
    for line in result.stdout.strip().splitlines():
        if line:
            rows.append(dict(zip(_OUTFMT_FIELDS, line.split("\t"))))
    return rows


def map_card_to_refseq(
    records: list[CARDRecord],
    db_path: str | Path,
    min_pident: float,
    min_qcov: float,
    max_evalue: float,
    num_threads: int = 1,
    chunk_size: int = 500,
) -> tuple[dict[str, BlastMapping], float]:
    """Map every CARD record onto the RefSeq BLAST database.

    Runs tblastn in chunks of `chunk_size` queries (rather than one call over
    all records) purely so progress is visible on stdout as chunks complete --
    a single call over thousands of queries gives no OS-visible signal of how
    far through the search it is until the whole thing exits.

    Args:
        records: CARD records to map (from card_parser.load_card_dataset).
        db_path: Prefix of a nucleotide BLAST database (from build_blast_database).
        min_pident: Minimum percent identity for a hit to count as mapped.
        min_qcov: Minimum percent query coverage for a hit to count as mapped.
        max_evalue: Maximum e-value, both for tblastn's native cutoff and the
            post-hoc filter below.
        num_threads: tblastn's -num_threads per chunk. See run_tblastn.
        chunk_size: Queries per tblastn invocation.

    Returns:
        Tuple of:
        - Dict mapping aro_accession -> BlastMapping (one entry per input
          record; unmapped entries have mapped=False and all-None fields).
        - Percent of records successfully mapped, as a reportable coverage
          number (CLAUDE.md: "record the actual coverage achieved").
    """
    hits_by_query: dict[str, list[dict[str, str]]] = {}
    num_chunks = (len(records) + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        chunk = records[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp:
            query_path = Path(tmp.name)
        _write_query_fasta(chunk, query_path)

        try:
            hits = run_tblastn(query_path, db_path, max_evalue, num_threads=num_threads)
        finally:
            query_path.unlink()

        for hit in hits:
            hits_by_query.setdefault(hit["qseqid"], []).append(hit)

        print(f"tblastn: {min((chunk_idx + 1) * chunk_size, len(records))}/{len(records)} queries done")

    results: dict[str, BlastMapping] = {}
    for record in records:
        candidates = hits_by_query.get(record.aro_accession, [])
        best = _pick_best_hit(candidates, min_pident, min_qcov)
        if best is None:
            results[record.aro_accession] = BlastMapping(
                aro_accession=record.aro_accession, mapped=False,
            )
        else:
            sstart, send = int(best["sstart"]), int(best["send"])
            results[record.aro_accession] = BlastMapping(
                aro_accession=record.aro_accession,
                mapped=True,
                replicon_accession=_strip_seqid_prefix(best["sseqid"]),
                start=min(sstart, send),
                end=max(sstart, send),
                strand=best["sstrand"],
                pident=float(best["pident"]),
                qcov=float(best["qcovs"]),
                evalue=float(best["evalue"]),
            )

    mapped_count = sum(1 for r in results.values() if r.mapped)
    coverage_pct = 100.0 * mapped_count / len(records) if records else 0.0
    return results, coverage_pct


def _pick_best_hit(
    candidates: list[dict[str, str]], min_pident: float, min_qcov: float,
) -> Optional[dict[str, str]]:
    """Return the highest-bitscore hit passing the identity/coverage thresholds."""
    passing = [c for c in candidates if float(c["pident"]) >= min_pident and float(c["qcovs"]) >= min_qcov]
    if not passing:
        return None
    return max(passing, key=lambda c: float(c["bitscore"]))


def _strip_seqid_prefix(sseqid: str) -> str:
    """Strip makeblastdb -parse_seqids's 'ref|...|' wrapping down to the bare accession.

    E.g. 'ref|NC_000913.3|' -> 'NC_000913.3'.
    """
    parts = sseqid.split("|")
    return parts[1] if len(parts) >= 2 and parts[0] == "ref" else sseqid
