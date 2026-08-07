"""Smoke tests for TA-proximity distance categorization (ta_proximity.py)."""

from src.data.blast_runner import BlastHit
from src.data.ta_proximity import TAProximityResult, compute_ta_proximity
from src.data.tadb_parser import TADBLocus


def _make_hit(aro_accession: str, replicon_accession: str, start: int, end: int) -> BlastHit:
    return BlastHit(
        aro_accession=aro_accession,
        replicon_accession=replicon_accession,
        start=start,
        end=end,
        strand="plus",
        percent_identity=100.0,
        evalue=0.0,
        bitscore=500.0,
        query_coverage=100.0,
    )


def _make_locus(locus_id: str, replicon_accession: str, start: int, end: int) -> TADBLocus:
    return TADBLocus(
        locus_id=locus_id,
        locus_type="toxin",
        confidence="exp",
        protein_accession="WP_000000000.1",
        replicon_accession=replicon_accession,
        start=start,
        end=end,
        strand="+",
        organism="Test organism",
    )


class TestComputeTaProximity:
    def test_real_distance_when_same_replicon(self):
        # Hit on NC_000913 at 1000-2000; nearest TA locus at 2500-2600 -- gap is 500bp.
        hits = [_make_hit("ARO:1", "NC_000913.3", 1000, 2000)]
        loci = [_make_locus("T1", "NC_000913", 2500, 2600)]

        results = compute_ta_proximity(hits, loci, all_aro_accessions=["ARO:1"])

        assert results == [
            TAProximityResult(
                aro_accession="ARO:1", category="distance", distance_bp=500, nearest_locus_id="T1"
            )
        ]

    def test_zero_distance_when_overlapping(self):
        hits = [_make_hit("ARO:1", "NC_000913.3", 1000, 2000)]
        loci = [_make_locus("T1", "NC_000913", 1500, 2500)]

        results = compute_ta_proximity(hits, loci, all_aro_accessions=["ARO:1"])

        assert results[0].category == "distance"
        assert results[0].distance_bp == 0

    def test_nearest_locus_chosen_among_multiple(self):
        hits = [_make_hit("ARO:1", "NC_000913.3", 1000, 2000)]
        loci = [
            _make_locus("T_far", "NC_000913", 10000, 10100),
            _make_locus("T_near", "NC_000913", 2100, 2200),
        ]

        results = compute_ta_proximity(hits, loci, all_aro_accessions=["ARO:1"])

        assert results[0].nearest_locus_id == "T_near"
        assert results[0].distance_bp == 100

    def test_no_ta_locus_when_replicon_has_none(self):
        hits = [_make_hit("ARO:1", "NC_000913.3", 1000, 2000)]
        loci = [_make_locus("T1", "OTHER_REPLICON", 1000, 2000)]

        results = compute_ta_proximity(hits, loci, all_aro_accessions=["ARO:1"])

        assert results == [TAProximityResult(aro_accession="ARO:1", category="no_ta_locus")]

    def test_unknown_when_no_blast_hit(self):
        results = compute_ta_proximity([], [], all_aro_accessions=["ARO:1"])
        assert results == [TAProximityResult(aro_accession="ARO:1", category="unknown")]

    def test_version_stripping_matches_across_sources(self):
        # BlastHit carries CARD's versioned accession; TADBLocus is unversioned.
        hits = [_make_hit("ARO:1", "AL123456.3", 100, 200)]
        loci = [_make_locus("T1", "AL123456", 300, 400)]

        results = compute_ta_proximity(hits, loci, all_aro_accessions=["ARO:1"])

        assert results[0].category == "distance"
        assert results[0].distance_bp == 100

    def test_different_replicon_never_compared_even_if_close(self):
        # Same organism, different replicon (e.g. chromosome vs. plasmid) --
        # must never be treated as same-replicon regardless of numeric coords.
        hits = [_make_hit("ARO:1", "NC_000913.3", 1000, 2000)]
        loci = [_make_locus("T1", "NC_000914", 1000, 2000)]  # different accession

        results = compute_ta_proximity(hits, loci, all_aro_accessions=["ARO:1"])

        assert results[0].category == "no_ta_locus"

    def test_all_accessions_present_even_mixed_categories(self):
        hits = [_make_hit("ARO:1", "NC_000913.3", 1000, 2000)]
        loci = [_make_locus("T1", "NC_000913", 5000, 5100)]

        results = compute_ta_proximity(
            hits, loci, all_aro_accessions=["ARO:1", "ARO:2", "ARO:3"]
        )

        by_aro = {r.aro_accession: r for r in results}
        assert len(results) == 3
        assert by_aro["ARO:1"].category == "distance"
        assert by_aro["ARO:2"].category == "unknown"  # no hit, no BLAST success
        assert by_aro["ARO:3"].category == "unknown"

    def test_unknown_and_no_ta_locus_never_conflated(self):
        # ARO:1 BLAST-mapped to a replicon with no TA locus (no_ta_locus);
        # ARO:2 never BLAST-mapped at all (unknown). Distinct categories.
        hits = [_make_hit("ARO:1", "NC_000913.3", 1000, 2000)]
        loci: list[TADBLocus] = []

        results = compute_ta_proximity(hits, loci, all_aro_accessions=["ARO:1", "ARO:2"])
        by_aro = {r.aro_accession: r for r in results}

        assert by_aro["ARO:1"].category == "no_ta_locus"
        assert by_aro["ARO:2"].category == "unknown"
        assert by_aro["ARO:1"].category != by_aro["ARO:2"].category
