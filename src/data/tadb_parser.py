"""Parse TADB 3.0 FASTA headers into structured TA-locus records.

Step 2 of the V2 TA-proximity pipeline (see CLAUDE.md's "TA-Proximity
Pipeline" section): TADB 3.0's headers are already RefSeq-anchored
(`replicon:coords`), so no BLAST step is needed here, unlike blast_runner.py's
CARD-to-RefSeq mapping. Downstream, ta_proximity.py joins this module's
output against blast_runner.py's BlastMapping records by replicon accession
to compute same-replicon bp distance.

Field names deliberately mirror blast_runner.BlastMapping (replicon_accession,
start, end, strand) so ta_proximity.py can treat both uniformly.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO

logger = logging.getLogger(__name__)

# Header format: >{locus_id} {protein_accession} {replicon}:{c?}{coord}-{coord} [{organism}]
# The "c" prefix (e.g. "c946611-946345") marks the complement (minus) strand;
# TADB's own accessions never carry a version suffix (e.g. "NC_000915", not
# "NC_000915.1"). Coordinate order isn't reliably low-then-high even within
# one strand, so parse_tadb_fasta always takes min()/max() of the pair.
_HEADER_RE = re.compile(
    r"^(?P<locus_id>[A-Z]+\d+)\s+(?P<protein_accession>\S+)\s+"
    r"(?P<replicon_accession>[^:\s]+):(?P<strand_marker>c?)(?P<coord1>\d+)-(?P<coord2>\d+)"
    r"\s+\[(?P<organism>[^\]]+)\]"
)


@dataclass
class TALocus:
    """A single TADB 3.0 toxin or antitoxin locus.

    Args:
        locus_id: TADB's local identifier (e.g. 'T28', 'AT28').
        locus_type: 'toxin' or 'antitoxin', derived from locus_id's prefix.
        protein_accession: RefSeq protein accession (e.g. 'WP_000916169.1').
        replicon_accession: RefSeq nucleotide accession, no version suffix
            (e.g. 'NC_000915').
        start: Lower genomic coordinate on the replicon.
        end: Higher genomic coordinate on the replicon.
        strand: 'plus' or 'minus'.
        organism: Source organism string from the FASTA header.
        source: 'exp' (experimentally validated) or 'pre' (in silico
            predicted) -- which TADB file this locus came from.
    """

    locus_id: str
    locus_type: str
    protein_accession: str
    replicon_accession: str
    start: int
    end: int
    strand: str
    organism: str
    source: str


def _locus_type_from_id(locus_id: str) -> str:
    """Derive 'toxin'/'antitoxin' from a locus_id's prefix (e.g. 'AT28' -> 'antitoxin').

    Checked in this order since 'AT' would otherwise also match a 'T' prefix check.
    """
    if locus_id.startswith("AT"):
        return "antitoxin"
    return "toxin"


def parse_tadb_fasta(fasta_path: str | Path, source: str) -> list[TALocus]:
    """Parse one TADB 3.0 FASTA file into TALocus records.

    Headers that don't match the expected format (missing organism brackets,
    no replicon:coords field, etc.) are logged as a warning and skipped --
    TADB isn't curated as strictly as CARD, so a parse failure on one record
    shouldn't abort the whole file.

    Args:
        fasta_path: Path to a type_II_{T,AT}_{exp,pre}.fas file.
        source: 'exp' or 'pre', recorded on every parsed TALocus.

    Returns:
        List of successfully parsed TALocus records (malformed headers omitted).
    """
    loci: list[TALocus] = []
    skipped = 0

    for record in SeqIO.parse(str(fasta_path), "fasta"):
        match = _HEADER_RE.match(record.description.strip())
        if not match:
            logger.warning(
                "Skipping malformed TADB header in %s: '%s'", fasta_path, record.description
            )
            skipped += 1
            continue

        fields = match.groupdict()
        coord1, coord2 = int(fields["coord1"]), int(fields["coord2"])
        loci.append(TALocus(
            locus_id=fields["locus_id"],
            locus_type=_locus_type_from_id(fields["locus_id"]),
            protein_accession=fields["protein_accession"],
            replicon_accession=fields["replicon_accession"],
            start=min(coord1, coord2),
            end=max(coord1, coord2),
            strand="minus" if fields["strand_marker"] == "c" else "plus",
            organism=fields["organism"].strip(),
            source=source,
        ))

    if skipped:
        logger.warning("Skipped %d malformed header(s) in %s", skipped, fasta_path)
    return loci
