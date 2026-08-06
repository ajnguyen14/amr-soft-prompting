"""Compute TA-locus proximity categories for CARD ARO accessions.

Steps 3-4 of the V2 TA-proximity pipeline (see CLAUDE.md's "TA-Proximity
Pipeline" section): joins blast_runner.py's CARD-to-RefSeq coordinate
mappings against tadb_parser.py's TA-locus coordinates to compute
same-replicon bp distance, then buckets those distances into a fixed
categorical vocabulary suitable for nn.Embedding (matching the existing
mechanism/drug_class conditioning pattern in soft_prompt.py).

RefSeq accessions carry a version suffix (e.g. 'NC_000913.3', confirmed
against downloaded genome FASTAs) while TADB's never do (e.g. 'NC_000913',
per tadb_parser.py's own docstring) -- both are normalized to the
version-less base accession before comparing, so the same physical replicon
is recognized as a match regardless of which side attached (or omitted) a
version.

Distance is defined as the gap between the two features' genomic intervals
(0 if they overlap), not point-to-point, since both a CARD hit and a TA
locus span a start-end range rather than a single coordinate.
"""

import bisect
from dataclasses import dataclass
from typing import Optional

from src.data.blast_runner import BlastMapping
from src.data.tadb_parser import TALocus

# Vocabulary entries for CLAUDE.md's data-quality-gap ('unknown') vs.
# real-signal ('no_ta_locus') distinction -- neither is ever an actual
# distance value, so both are kept out of the bin-derived labels below.
UNKNOWN_CATEGORY = "unknown"
NO_TA_LOCUS_CATEGORY = "no_ta_locus"


@dataclass
class ProximityResult:
    """TA-proximity result for one CARD ARO accession.

    Args:
        aro_accession: ARO ontology accession (e.g. 'ARO:3002999').
        mapped: Whether Step 1 (BLAST) placed this accession on RefSeq.
            Mirrors BlastMapping.mapped -- False means the accession belongs
            in the 'unknown' vocabulary bucket (a data-quality gap, not a
            biological signal).
        distance_bp: Same-replicon bp distance to the nearest TA locus, or
            None if unmapped, or mapped but no TA locus exists on that
            replicon ('no_ta_locus' -- a real biological signal, distinct
            from unmapped).
        nearest_locus_id: TADB locus_id of the nearest TA locus, or None
            unless distance_bp is set. Diagnostic only, not part of the
            categorical vocabulary.
        nearest_locus_source: 'exp' or 'pre', the TADB file the nearest
            locus came from, or None unless distance_bp is set. Diagnostic
            only.
    """

    aro_accession: str
    mapped: bool
    distance_bp: Optional[int] = None
    nearest_locus_id: Optional[str] = None
    nearest_locus_source: Optional[str] = None


def _normalize_replicon(accession: str) -> str:
    """Strip a RefSeq version suffix down to the base replicon accession.

    E.g. 'NC_000913.3' -> 'NC_000913'. TADB accessions never carry a version
    suffix to begin with, so this is a no-op for those inputs -- safe to
    apply uniformly to both sides of the join.

    Args:
        accession: A replicon accession, with or without a version suffix.

    Returns:
        The accession with any trailing '.<version>' removed.
    """
    return accession.split(".")[0]


def _group_loci_by_replicon(ta_loci: list[TALocus]) -> dict[str, list[TALocus]]:
    """Bucket TA loci by normalized replicon accession for fast lookup.

    Args:
        ta_loci: Combined list of TALocus records (both 'exp' and 'pre'
            sources -- see tadb_parser.parse_tadb_fasta).

    Returns:
        Dict mapping normalized replicon accession -> list of TALocus on
        that replicon.
    """
    by_replicon: dict[str, list[TALocus]] = {}
    for locus in ta_loci:
        key = _normalize_replicon(locus.replicon_accession)
        by_replicon.setdefault(key, []).append(locus)
    return by_replicon


def _interval_gap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    """bp gap between two closed genomic intervals, 0 if they overlap.

    Args:
        start_a: Lower coordinate of the first interval.
        end_a: Upper coordinate of the first interval.
        start_b: Lower coordinate of the second interval.
        end_b: Upper coordinate of the second interval.

    Returns:
        The gap in bp between the two intervals, or 0 if they overlap or
        touch.
    """
    return max(0, max(start_a, start_b) - min(end_a, end_b))


def _nearest_locus(
    mapping: BlastMapping, loci_by_replicon: dict[str, list[TALocus]]
) -> Optional[tuple[int, TALocus]]:
    """Find the closest TA locus on the same replicon as a BLAST mapping.

    Args:
        mapping: A mapped BlastMapping (caller must check mapping.mapped
            first -- this assumes start/end/replicon_accession are set).
        loci_by_replicon: Output of _group_loci_by_replicon.

    Returns:
        Tuple of (distance_bp, nearest TALocus), or None if no TA locus
        exists on this replicon at all.
    """
    replicon = _normalize_replicon(mapping.replicon_accession)
    candidates = loci_by_replicon.get(replicon)
    if not candidates:
        return None

    best_locus = min(
        candidates,
        key=lambda locus: _interval_gap(mapping.start, mapping.end, locus.start, locus.end),
    )
    best_distance = _interval_gap(mapping.start, mapping.end, best_locus.start, best_locus.end)
    return best_distance, best_locus


def compute_same_replicon_distances(
    blast_mappings: dict[str, BlastMapping], ta_loci: list[TALocus]
) -> dict[str, ProximityResult]:
    """Compute same-replicon TA-locus distance for every CARD ARO accession.

    Step 3 of the V2 TA-proximity pipeline. Unmapped accessions
    (BlastMapping.mapped is False) are recorded with mapped=False and
    distance_bp=None -- categorize() maps this to the 'unknown' vocabulary
    bucket. Mapped accessions with no TA locus on their replicon also get
    distance_bp=None, but mapped=True -- categorize() maps this to
    'no_ta_locus' instead, keeping the two data-quality-gap-vs-real-signal
    cases distinct per CLAUDE.md.

    Args:
        blast_mappings: aro_accession -> BlastMapping, from
            blast_runner.map_card_to_refseq.
        ta_loci: Combined TALocus list (both 'exp' and 'pre'), from
            tadb_parser.parse_tadb_fasta.

    Returns:
        Dict mapping aro_accession -> ProximityResult, one entry per input
        mapping.
    """
    loci_by_replicon = _group_loci_by_replicon(ta_loci)

    results: dict[str, ProximityResult] = {}
    for aro_accession, mapping in blast_mappings.items():
        if not mapping.mapped:
            results[aro_accession] = ProximityResult(aro_accession=aro_accession, mapped=False)
            continue

        nearest = _nearest_locus(mapping, loci_by_replicon)
        if nearest is None:
            results[aro_accession] = ProximityResult(aro_accession=aro_accession, mapped=True)
        else:
            distance_bp, locus = nearest
            results[aro_accession] = ProximityResult(
                aro_accession=aro_accession,
                mapped=True,
                distance_bp=distance_bp,
                nearest_locus_id=locus.locus_id,
                nearest_locus_source=locus.source,
            )
    return results


def build_category_vocab(bin_edges: list[int]) -> list[str]:
    """Build the ordered TA-proximity categorical vocabulary from bin edges.

    Args:
        bin_edges: Ascending bp thresholds separating real-distance bins
            (e.g. [1000, 10000, 100000]), set from the actual distance
            histogram per CLAUDE.md -- not chosen a priori, and read from
            config rather than hardcoded here.

    Returns:
        Ordered vocabulary list: [UNKNOWN_CATEGORY, NO_TA_LOCUS_CATEGORY,
        then one entry per real-distance bin]. List index doubles as the
        nn.Embedding row index for that category.
    """
    vocab = [UNKNOWN_CATEGORY, NO_TA_LOCUS_CATEGORY]

    lower = 0
    for edge in bin_edges:
        vocab.append(f"{lower}_{edge}bp")
        lower = edge
    vocab.append(f"gte_{lower}bp")

    return vocab


def categorize(results: dict[str, ProximityResult], bin_edges: list[int]) -> dict[str, str]:
    """Assign each ProximityResult to a category from the vocabulary.

    Step 4 of the V2 TA-proximity pipeline.

    Args:
        results: aro_accession -> ProximityResult, from
            compute_same_replicon_distances.
        bin_edges: Same ascending bp thresholds passed to
            build_category_vocab -- must match, or the categories returned
            here won't line up with that vocabulary's indices.

    Returns:
        Dict mapping aro_accession -> category string (one of
        build_category_vocab(bin_edges)'s entries).
    """
    vocab = build_category_vocab(bin_edges)

    categories: dict[str, str] = {}
    for aro_accession, result in results.items():
        if not result.mapped:
            categories[aro_accession] = UNKNOWN_CATEGORY
        elif result.distance_bp is None:
            categories[aro_accession] = NO_TA_LOCUS_CATEGORY
        else:
            bin_idx = bisect.bisect_right(bin_edges, result.distance_bp)
            categories[aro_accession] = vocab[2 + bin_idx]
    return categories
