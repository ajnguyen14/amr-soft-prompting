"""Parse TADB 3.0 Type II toxin/antitoxin FASTA headers into locus records.

Only the toxin/antitoxin protein files are parsed (`type_II_{T,AT}_{exp,pre}.fas`)
-- the Type II regulator files are intentionally unused (CLAUDE.md TA-Proximity
Pipeline: redundant with toxin/antitoxin coordinates for the same locus).
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Header format: >{locus_id} {protein_accession} {replicon}:[c]{start}-{end} [{organism}]
# A leading "c" before the coordinates marks the minus/complement strand, in which
# case TADB lists the coordinates in reverse (larger-first) order.
# Coordinates are normally plain integers, but at least one row in
# type_II_AT_pre.fas ("2e+06-2000257") has a coordinate in scientific notation --
# an upstream TADB export artifact -- so the coordinate groups accept that too.
_HEADER_RE = re.compile(
    r"^>(?P<locus_id>\S+)\s+(?P<protein_accession>\S+)\s+"
    r"(?P<replicon_accession>[^:\s]+):(?P<strand_marker>c?)"
    r"(?P<coord1>[\d.eE+]+)-(?P<coord2>[\d.eE+]+)\s+\[(?P<organism>.+)\]$"
)

# (filename, locus_type, confidence) for the four TADB 3.0 Type II source files.
_SOURCE_FILES = (
    ("type_II_T_exp.fas", "toxin", "exp"),
    ("type_II_AT_exp.fas", "antitoxin", "exp"),
    ("type_II_T_pre.fas", "toxin", "pre"),
    ("type_II_AT_pre.fas", "antitoxin", "pre"),
)


@dataclass(frozen=True)
class TADBLocus:
    """A single TADB 3.0 Type II toxin or antitoxin locus.

    Args:
        locus_id: TADB's own locus identifier (e.g. 'T28', 'AT20007').
        locus_type: 'toxin' or 'antitoxin'.
        confidence: 'exp' (experimentally validated, 403 pairs) or 'pre'
            (in silico predicted, larger and lower confidence).
        protein_accession: NCBI protein accession for this component.
        replicon_accession: RefSeq/GenBank replicon accession, unversioned --
            TADB headers never carry a version suffix, unlike CARD's DNA
            Accession field (see CLAUDE.md TA-Proximity Pipeline).
        start: 1-based start coordinate on the replicon (always <= end,
            regardless of strand).
        end: 1-based end coordinate on the replicon.
        strand: '+' or '-'.
        organism: Organism name as recorded in the header.
    """

    locus_id: str
    locus_type: str
    confidence: str
    protein_accession: str
    replicon_accession: str
    start: int
    end: int
    strand: str
    organism: str


def _parse_coord(token: str) -> int:
    """Parse a header coordinate token, tolerating scientific notation.

    Args:
        token: Raw coordinate text from a TADB header (e.g. '946611' or,
            for one known malformed row in type_II_AT_pre.fas, '2e+06').

    Returns:
        The coordinate as an integer.
    """
    return int(float(token))


def parse_tadb_fasta(path: str | Path, locus_type: str, confidence: str) -> list[TADBLocus]:
    """Parse one TADB 3.0 Type II FASTA file into locus records.

    Args:
        path: Path to a `type_II_{T,AT}_{exp,pre}.fas` file.
        locus_type: 'toxin' or 'antitoxin' -- which component this file holds.
        confidence: 'exp' or 'pre', matching the source filename.

    Returns:
        One TADBLocus per well-formed FASTA header in the file. Malformed
        headers are skipped with a warning rather than raising, matching
        card_parser.py's resilience convention.
    """
    if locus_type not in ("toxin", "antitoxin"):
        raise ValueError(f"locus_type must be 'toxin' or 'antitoxin', got {locus_type!r}")
    if confidence not in ("exp", "pre"):
        raise ValueError(f"confidence must be 'exp' or 'pre', got {confidence!r}")

    loci: list[TADBLocus] = []
    skipped = 0

    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            if not line.startswith(">"):
                continue

            match = _HEADER_RE.match(line.rstrip("\n"))
            if match is None:
                logger.warning("%s:%d: unrecognized TADB header format: %r", path, line_num, line)
                skipped += 1
                continue

            coord1 = _parse_coord(match.group("coord1"))
            coord2 = _parse_coord(match.group("coord2"))
            loci.append(
                TADBLocus(
                    locus_id=match.group("locus_id"),
                    locus_type=locus_type,
                    confidence=confidence,
                    protein_accession=match.group("protein_accession"),
                    replicon_accession=match.group("replicon_accession"),
                    start=min(coord1, coord2),
                    end=max(coord1, coord2),
                    strand="-" if match.group("strand_marker") == "c" else "+",
                    organism=match.group("organism"),
                )
            )

    logger.info(
        "Loaded %d %s/%s TADB loci (%d skipped) from %s",
        len(loci),
        locus_type,
        confidence,
        skipped,
        path,
    )
    return loci


def load_all_tadb_loci(raw_data_dir: str | Path) -> list[TADBLocus]:
    """Load and combine all four TADB 3.0 Type II source files.

    Args:
        raw_data_dir: Directory containing the four
            `type_II_{T,AT}_{exp,pre}.fas` files (CLAUDE.md paths convention:
            `data/raw/`).

    Returns:
        Combined list of TADBLocus records across all four files.
    """
    raw_data_dir = Path(raw_data_dir)
    loci: list[TADBLocus] = []
    for filename, locus_type, confidence in _SOURCE_FILES:
        loci.extend(parse_tadb_fasta(raw_data_dir / filename, locus_type, confidence))
    return loci


def replicon_accessions(loci: list[TADBLocus]) -> set[str]:
    """Collect the distinct replicon accessions referenced by a set of loci.

    Used to build the CARD/TADB accession-intersection prefilter (CLAUDE.md
    TA-Proximity Pipeline Step 1): only replicons in this set need to be
    fetched from RefSeq and BLASTed against.

    Args:
        loci: TADBLocus records, e.g. from load_all_tadb_loci.

    Returns:
        Set of unversioned replicon accessions.
    """
    return {locus.replicon_accession for locus in loci}
