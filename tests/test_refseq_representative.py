"""Smoke tests for RefSeq representative-accession selection (refseq_representative.py)."""

import json
from pathlib import Path

import pytest

from src.data.refseq_representative import (
    AroRepresentativeMapping,
    AroTaxonomyRecord,
    RepresentativeAccession,
    get_fetch_accession_list,
    load_aro_taxonomy_records,
    map_aro_to_representative,
    select_representative_accessions,
)

# ---------------------------------------------------------------------------
# Path to the real card.json -- integration test skipped if absent.
# ---------------------------------------------------------------------------
_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
_CARD_JSON_PATH = _RAW_DIR / "card.json"
_skip_no_data = pytest.mark.skipif(
    not _CARD_JSON_PATH.exists(),
    reason="card.json not present (expected on CPU/GPU server)",
)


def _seq_entry(dna_accession: str, taxonomy_id: str, taxonomy_name: str) -> dict:
    """Build a minimal model_sequences.sequence.<id> block."""
    return {
        "dna_sequence": {"accession": dna_accession},
        "NCBI_taxonomy": {
            "NCBI_taxonomy_id": taxonomy_id,
            "NCBI_taxonomy_name": taxonomy_name,
        },
    }


def _card_entry(aro_accession_bare: str, seq: dict) -> dict:
    """Build a minimal card.json top-level entry."""
    return {
        "ARO_accession": aro_accession_bare,
        "model_sequences": {"sequence": {"1": seq}},
    }


# ---------------------------------------------------------------------------
# Fixture: minimal in-memory card.json --
# - two ARO entries share taxonomy_id "100" but have different DNA accessions
#   (one accession appears twice, so it's the clear most-common winner)
# - two ARO entries share taxonomy_id "200" with two *different* accessions,
#   each appearing once (a tie, broken lexicographically)
# - one entry has taxonomy_id "300" alone (a singleton group)
# - one entry is missing NCBI_taxonomy entirely (skipped)
# ---------------------------------------------------------------------------

MINIMAL_CARD = {
    "1": _card_entry("3000001", _seq_entry("AA000001.1", "100", "Organism A")),
    "2": _card_entry("3000002", _seq_entry("AA000001.1", "100", "Organism A")),
    "3": _card_entry("3000003", _seq_entry("BB000002.1", "100", "Organism A")),
    "4": _card_entry("3000004", _seq_entry("CC000003.1", "200", "Organism B")),
    "5": _card_entry("3000005", _seq_entry("DD000004.1", "200", "Organism B")),
    "6": _card_entry("3000006", _seq_entry("EE000005.1", "300", "Organism C")),
    "7": {
        "ARO_accession": "3000007",
        "model_sequences": {"sequence": {"1": {"dna_sequence": {"accession": "FF000006.1"}}}},
    },
}


@pytest.fixture()
def minimal_card_json(tmp_path: Path) -> Path:
    card_file = tmp_path / "test_card.json"
    card_file.write_text(json.dumps(MINIMAL_CARD))
    return card_file


@pytest.fixture()
def minimal_records(minimal_card_json: Path) -> list[AroTaxonomyRecord]:
    return load_aro_taxonomy_records(minimal_card_json)


@pytest.fixture()
def minimal_representatives(
    minimal_records: list[AroTaxonomyRecord],
) -> list[RepresentativeAccession]:
    return select_representative_accessions(minimal_records)


# ---------------------------------------------------------------------------
# Unit tests: load_aro_taxonomy_records
# ---------------------------------------------------------------------------


class TestLoadAroTaxonomyRecords:
    def test_record_count_excludes_missing_taxonomy(self, minimal_records):
        # Entry "7" has no dna_sequence.accession or NCBI_taxonomy -- skipped.
        assert len(minimal_records) == 6

    def test_aro_accession_prefix_normalized(self, minimal_records):
        aros = {r.aro_accession for r in minimal_records}
        assert "ARO:3000001" in aros
        assert "3000001" not in aros

    def test_record_type(self, minimal_records):
        assert all(isinstance(r, AroTaxonomyRecord) for r in minimal_records)

    def test_fields_populated(self, minimal_records):
        rec = next(r for r in minimal_records if r.aro_accession == "ARO:3000001")
        assert rec.dna_accession == "AA000001.1"
        assert rec.taxonomy_id == "100"
        assert rec.taxonomy_name == "Organism A"


# ---------------------------------------------------------------------------
# Unit tests: select_representative_accessions
# ---------------------------------------------------------------------------


class TestSelectRepresentativeAccessions:
    def test_one_representative_per_taxonomy_group(self, minimal_representatives):
        taxonomy_ids = {r.taxonomy_id for r in minimal_representatives}
        assert taxonomy_ids == {"100", "200", "300"}
        assert len(minimal_representatives) == 3

    def test_clear_winner_selected_without_tie(self, minimal_representatives):
        group_100 = next(r for r in minimal_representatives if r.taxonomy_id == "100")
        assert group_100.representative_accession == "AA000001.1"
        assert group_100.group_size == 3
        assert group_100.candidate_accession_count == 2
        assert group_100.tie_broken is False

    def test_tie_broken_lexicographically(self, minimal_representatives):
        group_200 = next(r for r in minimal_representatives if r.taxonomy_id == "200")
        # CC000003.1 < DD000004.1 lexicographically, each appears once.
        assert group_200.representative_accession == "CC000003.1"
        assert group_200.tie_broken is True

    def test_singleton_group_not_tie_broken(self, minimal_representatives):
        group_300 = next(r for r in minimal_representatives if r.taxonomy_id == "300")
        assert group_300.representative_accession == "EE000005.1"
        assert group_300.candidate_accession_count == 1
        assert group_300.tie_broken is False

    def test_record_type(self, minimal_representatives):
        assert all(isinstance(r, RepresentativeAccession) for r in minimal_representatives)


# ---------------------------------------------------------------------------
# Unit tests: map_aro_to_representative
# ---------------------------------------------------------------------------


class TestMapAroToRepresentative:
    def test_one_mapping_per_record(self, minimal_records, minimal_representatives):
        mappings = map_aro_to_representative(minimal_records, minimal_representatives)
        assert len(mappings) == len(minimal_records)
        assert all(isinstance(m, AroRepresentativeMapping) for m in mappings)

    def test_used_own_accession_true_for_winner(self, minimal_records, minimal_representatives):
        mappings = map_aro_to_representative(minimal_records, minimal_representatives)
        m = next(m for m in mappings if m.aro_accession == "ARO:3000001")
        assert m.used_own_accession is True
        assert m.representative_accession == "AA000001.1"

    def test_used_own_accession_false_for_substituted(
        self, minimal_records, minimal_representatives
    ):
        mappings = map_aro_to_representative(minimal_records, minimal_representatives)
        # ARO:3000003 is BB000002.1, but group 100's representative is AA000001.1.
        m = next(m for m in mappings if m.aro_accession == "ARO:3000003")
        assert m.used_own_accession is False
        assert m.own_dna_accession == "BB000002.1"
        assert m.representative_accession == "AA000001.1"


# ---------------------------------------------------------------------------
# Unit tests: get_fetch_accession_list
# ---------------------------------------------------------------------------


class TestGetFetchAccessionList:
    def test_returns_sorted_deduplicated_list(self, minimal_representatives):
        fetch_list = get_fetch_accession_list(minimal_representatives)
        assert fetch_list == sorted(fetch_list)
        assert len(fetch_list) == len(set(fetch_list))

    def test_expected_accessions(self, minimal_representatives):
        fetch_list = get_fetch_accession_list(minimal_representatives)
        assert fetch_list == ["AA000001.1", "CC000003.1", "EE000005.1"]


# ---------------------------------------------------------------------------
# Integration test: full CARD dataset (skipped when data absent)
# ---------------------------------------------------------------------------


@_skip_no_data
class TestFullSelection:
    def test_expected_group_and_fetch_counts(self):
        # CARD broadstreet v4.0.1: 6404 ARO entries have both a DNA accession
        # and NCBI taxonomy info in card.json, grouping into 740 distinct
        # taxonomy IDs. Two groups happen to converge on the same
        # representative accession, so the deduplicated fetch list is 738.
        records = load_aro_taxonomy_records(_CARD_JSON_PATH)
        assert len(records) == 6404

        representatives = select_representative_accessions(records)
        assert len(representatives) == 740

        fetch_list = get_fetch_accession_list(representatives)
        assert len(fetch_list) == 738

    def test_substitution_rate_is_high_for_strain_diverse_species(self):
        # Large species groups in CARD (P. aeruginosa, K. pneumoniae, E. coli,
        # etc.) have nearly one distinct DNA accession per ARO entry, so
        # "most common" barely reduces substitution for them -- most of the
        # majority-substituted 5316/6404 total comes from these few large
        # groups, not a uniform spread across all 740 groups.
        records = load_aro_taxonomy_records(_CARD_JSON_PATH)
        representatives = select_representative_accessions(records)
        mappings = map_aro_to_representative(records, representatives)
        substituted = sum(1 for m in mappings if not m.used_own_accession)
        assert substituted == 5316

    def test_all_representative_accessions_are_versioned_card_accessions(self):
        records = load_aro_taxonomy_records(_CARD_JSON_PATH)
        representatives = select_representative_accessions(records)
        card_accessions = {r.dna_accession for r in records}
        assert all(
            r.representative_accession in card_accessions for r in representatives
        )
