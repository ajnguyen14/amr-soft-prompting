"""Compute same-replicon bp distance from CARD BLAST hits to the nearest
TADB TA locus (CLAUDE.md TA-Proximity Pipeline Step 3).

Only compares coordinates when a BlastHit's replicon and a TADBLocus's
replicon are the exact same accession (version-stripped, since BLAST hits
carry CARD's versioned accession while TADB's are already unversioned --
same mismatch card_tadb_matcher.py already handles) -- never across
different replicons/assemblies, even from the same organism, per CLAUDE.md.

Categorizes every CARD ARO accession into exactly one of:
  - a real same-replicon bp distance to the nearest TA locus
  - 'no_ta_locus' -- BLAST-mapped successfully, but no TA locus exists on
    that replicon (a real biological signal)
  - 'unknown' -- BLAST mapping failed entirely (Step 1), a data-quality gap,
    never conflated with 'no_ta_locus'

The categorical bin-edge embedding vocabulary (Step 4) is deliberately not
built here -- CLAUDE.md requires bin edges to be set from the actual
distance histogram once Steps 1-3 have run on the full dataset, not chosen
a priori. This module only produces the raw per-ARO distance/category
values; binning them into an nn.Embedding vocabulary is a follow-up once the
full BLAST job's real hits are available.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.data.blast_runner import BlastHit
from src.data.card_tadb_matcher import strip_accession_version
from src.data.tadb_parser import TADBLocus

logger = logging.getLogger(__name__)

_UNKNOWN = "unknown"
_NO_TA_LOCUS = "no_ta_locus"
_DISTANCE = "distance"


@dataclass(frozen=True)
class TAProximityResult:
    """Same-replicon TA-locus proximity for one CARD ARO accession.

    Args:
        aro_accession: CARD ARO accession.
        category: One of 'distance', 'no_ta_locus', or 'unknown'.
        distance_bp: Nearest same-replicon TA locus distance in bp, only set
            when category == 'distance'. None otherwise.
        nearest_locus_id: TADBLocus.locus_id of the nearest same-replicon TA
            locus, only set when category == 'distance'. None otherwise.
        used_own_accession: True if this ARO accession was BLASTed against
            its own recorded genome; False if against a substituted
            (different-strain) representative genome (see
            refseq_representative.AroRepresentativeMapping); None if not
            known (accession absent from the query universe passed in).
            Without this, a consumer of this result can't distinguish a
            distance/no_ta_locus computed against the gene's actual source
            genome from one computed against a different strain's assembly
            -- a real bias affecting 83% of CARD entries (docs/STATUS.md).
    """

    aro_accession: str
    category: str
    distance_bp: Optional[int] = None
    nearest_locus_id: Optional[str] = None
    used_own_accession: Optional[bool] = None


def _distance_to_locus(hit: BlastHit, locus: TADBLocus) -> int:
    """bp distance between a same-replicon BLAST hit and a TADB locus.

    Zero if the two intervals overlap; otherwise the gap between their
    nearest edges.

    Args:
        hit: A BlastHit already confirmed to share a replicon with locus.
        locus: A TADBLocus already confirmed to share a replicon with hit.

    Returns:
        Non-negative bp distance.
    """
    if hit.end < locus.start:
        return locus.start - hit.end
    if locus.end < hit.start:
        return hit.start - locus.end
    return 0  # overlapping intervals


def compute_ta_proximity(
    hits: list[BlastHit],
    tadb_loci: list[TADBLocus],
    all_aro_accessions: list[str],
    used_own_accession_by_aro: Optional[dict[str, bool]] = None,
) -> list[TAProximityResult]:
    """Classify every ARO accession's TA-locus proximity from BLAST hits.

    Args:
        hits: BlastHit list from
            blast_runner.blast_card_against_representatives -- one entry per
            ARO accession that BLAST-mapped successfully.
        tadb_loci: TADBLocus list, e.g. from tadb_parser.load_all_tadb_loci.
        all_aro_accessions: Every ARO accession Step 1 attempted to map (the
            full query set, not just the successes) -- accessions absent
            from `hits` are categorized 'unknown' rather than silently
            dropped, since the TA-proximity vocabulary needs an explicit
            value for every accession, not just the ones BLAST succeeded on.
        used_own_accession_by_aro: Optional dict from
            refseq_representative.AroRepresentativeMapping (aro_accession ->
            used_own_accession), used to populate
            TAProximityResult.used_own_accession so the substitution-genome
            bias is auditable on the final result, not just an intermediate
            Step 1 detail. An accession absent from this dict gets
            used_own_accession=None (unknown), not False.

    Returns:
        One TAProximityResult per accession in all_aro_accessions.
    """
    used_own_accession_by_aro = used_own_accession_by_aro or {}

    loci_by_replicon: dict[str, list[TADBLocus]] = {}
    for locus in tadb_loci:
        loci_by_replicon.setdefault(locus.replicon_accession, []).append(locus)

    hit_by_aro = {hit.aro_accession: hit for hit in hits}

    results: list[TAProximityResult] = []
    for aro_accession in all_aro_accessions:
        used_own = used_own_accession_by_aro.get(aro_accession)
        hit = hit_by_aro.get(aro_accession)
        if hit is None:
            results.append(
                TAProximityResult(
                    aro_accession=aro_accession, category=_UNKNOWN, used_own_accession=used_own
                )
            )
            continue

        replicon = strip_accession_version(hit.replicon_accession)
        same_replicon_loci = loci_by_replicon.get(replicon, [])
        if not same_replicon_loci:
            results.append(
                TAProximityResult(
                    aro_accession=aro_accession, category=_NO_TA_LOCUS, used_own_accession=used_own
                )
            )
            continue

        nearest = min(same_replicon_loci, key=lambda locus: _distance_to_locus(hit, locus))
        results.append(
            TAProximityResult(
                aro_accession=aro_accession,
                category=_DISTANCE,
                distance_bp=_distance_to_locus(hit, nearest),
                nearest_locus_id=nearest.locus_id,
                used_own_accession=used_own,
            )
        )

    n_distance = sum(1 for r in results if r.category == _DISTANCE)
    n_no_ta = sum(1 for r in results if r.category == _NO_TA_LOCUS)
    n_unknown = sum(1 for r in results if r.category == _UNKNOWN)
    logger.info(
        "TA-proximity categorized %d accessions: %d distance, %d no_ta_locus, %d unknown",
        len(results),
        n_distance,
        n_no_ta,
        n_unknown,
    )
    return results
