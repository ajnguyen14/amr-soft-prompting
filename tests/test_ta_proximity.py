"""Smoke tests for src/data/ta_proximity.py."""

from src.data.blast_runner import BlastMapping
from src.data.tadb_parser import TALocus
from src.data.ta_proximity import (
    NO_TA_LOCUS_CATEGORY,
    UNKNOWN_CATEGORY,
    build_category_vocab,
    categorize,
    compute_same_replicon_distances,
)


def _locus(replicon: str, start: int, end: int, locus_id: str = "T1", source: str = "exp") -> TALocus:
    return TALocus(
        locus_id=locus_id,
        locus_type="toxin",
        protein_accession="WP_000001.1",
        replicon_accession=replicon,
        start=start,
        end=end,
        strand="plus",
        organism="Escherichia coli",
        source=source,
    )


def _mapping(aro_accession: str, mapped: bool, replicon: str = None, start: int = None, end: int = None) -> BlastMapping:
    return BlastMapping(
        aro_accession=aro_accession,
        mapped=mapped,
        replicon_accession=replicon,
        start=start,
        end=end,
        strand="plus" if mapped else None,
        pident=99.0 if mapped else None,
        qcov=95.0 if mapped else None,
        evalue=1e-50 if mapped else None,
    )


def test_unmapped_accession_recorded_with_no_distance():
    mappings = {"ARO:1": _mapping("ARO:1", mapped=False)}
    results = compute_same_replicon_distances(mappings, ta_loci=[])

    assert results["ARO:1"].mapped is False
    assert results["ARO:1"].distance_bp is None


def test_mapped_accession_with_no_loci_on_replicon():
    mappings = {"ARO:1": _mapping("ARO:1", mapped=True, replicon="NC_000913.3", start=100, end=200)}
    loci = [_locus("NC_000042.1", start=1000, end=1100)]

    results = compute_same_replicon_distances(mappings, loci)

    assert results["ARO:1"].mapped is True
    assert results["ARO:1"].distance_bp is None


def test_accession_version_suffix_is_normalized_for_matching():
    # BLAST-mapped replicon carries a version suffix; the TADB locus's
    # replicon never does -- both must normalize to the same base accession
    # for the join to find this locus.
    mappings = {"ARO:1": _mapping("ARO:1", mapped=True, replicon="NC_000913.3", start=1000, end=1100)}
    loci = [_locus("NC_000913", start=5000, end=5100)]

    results = compute_same_replicon_distances(mappings, loci)

    assert results["ARO:1"].distance_bp == 3900
    assert results["ARO:1"].nearest_locus_id == "T1"


def test_overlapping_interval_has_zero_distance():
    mappings = {"ARO:1": _mapping("ARO:1", mapped=True, replicon="NC_000913", start=1000, end=2000)}
    loci = [_locus("NC_000913", start=1500, end=2500)]

    results = compute_same_replicon_distances(mappings, loci)

    assert results["ARO:1"].distance_bp == 0


def test_nearest_of_multiple_loci_on_same_replicon_is_chosen():
    mappings = {"ARO:1": _mapping("ARO:1", mapped=True, replicon="NC_000913", start=1000, end=1000)}
    loci = [
        _locus("NC_000913", start=5000, end=5000, locus_id="far"),
        _locus("NC_000913", start=1500, end=1500, locus_id="near"),
    ]

    results = compute_same_replicon_distances(mappings, loci)

    assert results["ARO:1"].nearest_locus_id == "near"
    assert results["ARO:1"].distance_bp == 500


def test_build_category_vocab_shape():
    vocab = build_category_vocab(bin_edges=[1000, 10000, 100000])

    assert vocab == [
        UNKNOWN_CATEGORY,
        NO_TA_LOCUS_CATEGORY,
        "0_1000bp",
        "1000_10000bp",
        "10000_100000bp",
        "gte_100000bp",
    ]


def test_categorize_assigns_all_three_bucket_types():
    mappings = {
        "unmapped": _mapping("unmapped", mapped=False),
        "no_locus": _mapping("no_locus", mapped=True, replicon="NC_1", start=100, end=200),
        "near": _mapping("near", mapped=True, replicon="NC_2", start=1000, end=1000),
        "far": _mapping("far", mapped=True, replicon="NC_2", start=500000, end=500000),
    }
    loci = [
        _locus("NC_2", start=1500, end=1500, locus_id="near_locus"),
        _locus("NC_2", start=200000, end=200000, locus_id="far_locus"),
    ]
    bin_edges = [1000, 10000, 100000]

    results = compute_same_replicon_distances(mappings, loci)
    categories = categorize(results, bin_edges)

    assert categories["unmapped"] == UNKNOWN_CATEGORY
    assert categories["no_locus"] == NO_TA_LOCUS_CATEGORY
    assert categories["near"] == "0_1000bp"
    assert categories["far"] == "gte_100000bp"
