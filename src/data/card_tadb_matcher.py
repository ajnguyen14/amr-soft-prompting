"""Match CARD DNA accessions against TADB 3.0 replicon accessions.

NOT PART OF THE LIVE PIPELINE -- superseded, kept for its accession-matching
logic and tests only. Neither scripts/run_blast_coordinate_mapping.py nor
scripts/run_ta_proximity.py imports match_card_to_tadb_replicons; the only
caller is this module's own test file. It was originally built as an
accession-intersection prefilter to scope the BLAST step to a bounded
replicon set, but that prefilter forced ~97.6% of CARD entries into
'unknown' regardless of whether they actually had a nearby TA locus,
conflating "BLAST not attempted" with the real 'no_ta_locus' signal
CLAUDE.md's vocabulary distinguishes (see docs/STATUS.md). The live pipeline
uses src/data/refseq_representative.py's organism-taxonomy grouping to scope
the BLAST step instead. strip_accession_version, defined here, IS still used
live -- by src/data/ta_proximity.py, for the unrelated purpose of comparing
BlastHit and TADBLocus replicon accessions.

This only tells us *which replicon* a CARD gene's source sequence lives on
-- CARD's own fmin/fmax fields are relative to its own excised gene
fragment, not the replicon, so they cannot supply real genomic coordinates
(confirmed empirically: fmax - fmin == len(sequence) for all 6404 CARD
records with numeric coordinates). BLAST against the matched replicon is
still required to place the gene within it.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from src.data.card_parser import parse_aro_index
from src.data.tadb_parser import TADBLocus

logger = logging.getLogger(__name__)


def strip_accession_version(accession: str) -> str:
    """Strip a GenBank/RefSeq version suffix, e.g. 'AL123456.3' -> 'AL123456'.

    Some older GenBank accessions carry no version suffix at all; those pass
    through unchanged. TADB replicon accessions never carry a version suffix
    (see tadb_parser.py), so in practice this only needs to run on the CARD
    side of a match.

    Args:
        accession: A GenBank/RefSeq accession, with or without a version suffix.

    Returns:
        The accession with any trailing '.<version>' removed.
    """
    return accession.rsplit(".", 1)[0] if "." in accession else accession


@dataclass(frozen=True)
class AccessionMatch:
    """One CARD ARO accession whose DNA Accession matches a TADB replicon.

    Args:
        aro_accession: CARD ARO accession (e.g. 'ARO:3002999').
        card_dna_accession: CARD's original, versioned DNA Accession (e.g.
            'AL123456.3') -- pinned for the later RefSeq fetch/BLAST step so
            results stay self-consistent with the exact sequence version
            CARD's protein was drawn from, rather than whatever the current
            live RefSeq version happens to be.
        base_accession: The version-stripped accession shared with TADB
            (e.g. 'AL123456').
    """

    aro_accession: str
    card_dna_accession: str
    base_accession: str


def match_card_to_tadb_replicons(
    aro_index_path: str | Path,
    tadb_loci: list[TADBLocus],
) -> list[AccessionMatch]:
    """Find CARD ARO accessions whose DNA Accession matches a TADB replicon.

    NOT PART OF THE LIVE PIPELINE -- see this module's docstring for why
    (superseded by refseq_representative.py's organism-taxonomy grouping).
    Exercised only by this module's own test file.

    Args:
        aro_index_path: Path to aro_index.tsv.
        tadb_loci: TADBLocus records, e.g. from tadb_parser.load_all_tadb_loci.

    Returns:
        One AccessionMatch per CARD ARO accession with a non-empty DNA
        Accession whose version-stripped base matches a TADB replicon
        accession. ARO accessions with no DNA Accession, or whose DNA
        Accession's base doesn't appear in tadb_loci, are excluded (they
        become 'unknown' downstream in the TA-proximity vocabulary, per
        CLAUDE.md -- but that encoding happens in ta_proximity.py, not here).
    """
    tadb_replicons = {locus.replicon_accession for locus in tadb_loci}

    aro_index = parse_aro_index(aro_index_path)
    matches: list[AccessionMatch] = []
    for aro_accession, row in aro_index.items():
        dna_accession = row.get("DNA Accession", "").strip()
        if not dna_accession:
            continue

        base_accession = strip_accession_version(dna_accession)
        if base_accession in tadb_replicons:
            matches.append(
                AccessionMatch(
                    aro_accession=aro_accession,
                    card_dna_accession=dna_accession,
                    base_accession=base_accession,
                )
            )

    logger.info(
        "Matched %d/%d CARD ARO accessions to a TADB replicon",
        len(matches),
        len(aro_index),
    )
    return matches
