"""Smoke tests for the RefSeq representative-accession fetcher (refseq_fetch.py).

All tests mock Bio.Entrez.efetch -- no real network calls, per the project's
lightweight-smoke-test philosophy (tests must be fast and not depend on an
external service being reachable).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.refseq_fetch import FetchResult, fetch_representative_sequences


def _make_mock_efetch(fasta_by_accession: dict[str, str], fail_accessions: set[str] = frozenset()):
    """Build a Bio.Entrez.efetch replacement backed by an in-memory FASTA map."""

    def efetch(db, id, rettype, retmode):
        if id in fail_accessions:
            raise IOError(f"simulated NCBI failure for {id}")
        handle = MagicMock()
        handle.read.return_value = fasta_by_accession[id]
        handle.__enter__.return_value = handle
        handle.__exit__.return_value = False
        return handle

    return efetch


# ---------------------------------------------------------------------------
# Unit tests: fetch_representative_sequences
# ---------------------------------------------------------------------------


class TestFetchRepresentativeSequences:
    def test_writes_one_file_per_accession(self, tmp_path: Path):
        fasta_map = {
            "AA000001.1": ">AA000001.1 test\nACGT\n",
            "BB000002.1": ">BB000002.1 test\nTTTT\n",
        }
        with patch("src.data.refseq_fetch.Entrez.efetch", side_effect=_make_mock_efetch(fasta_map)), \
             patch("src.data.refseq_fetch.time.sleep"):
            result = fetch_representative_sequences(
                accessions=list(fasta_map),
                email="test@example.com",
                output_dir=tmp_path,
            )

        assert (tmp_path / "AA000001.1.fasta").read_text() == fasta_map["AA000001.1"]
        assert (tmp_path / "BB000002.1.fasta").read_text() == fasta_map["BB000002.1"]
        assert result.succeeded == ["AA000001.1", "BB000002.1"]
        assert result.failed == {}

    def test_sets_entrez_email_and_api_key(self, tmp_path: Path):
        from src.data.refseq_fetch import Entrez

        fasta_map = {"AA000001.1": ">AA000001.1 test\nACGT\n"}
        with patch("src.data.refseq_fetch.Entrez.efetch", side_effect=_make_mock_efetch(fasta_map)), \
             patch("src.data.refseq_fetch.time.sleep"):
            fetch_representative_sequences(
                accessions=list(fasta_map),
                email="test@example.com",
                output_dir=tmp_path,
                api_key="fake-key-123",
            )

        assert Entrez.email == "test@example.com"
        assert Entrez.api_key == "fake-key-123"

    def test_failed_accession_recorded_not_written(self, tmp_path: Path):
        fasta_map = {"AA000001.1": ">AA000001.1 test\nACGT\n", "BB000002.1": ""}
        with patch(
            "src.data.refseq_fetch.Entrez.efetch",
            side_effect=_make_mock_efetch(fasta_map, fail_accessions={"BB000002.1"}),
        ), patch("src.data.refseq_fetch.time.sleep"):
            result = fetch_representative_sequences(
                accessions=["AA000001.1", "BB000002.1"],
                email="test@example.com",
                output_dir=tmp_path,
            )

        assert result.succeeded == ["AA000001.1"]
        assert "BB000002.1" in result.failed
        assert not (tmp_path / "BB000002.1.fasta").exists()

    def test_existing_file_skipped_not_refetched(self, tmp_path: Path):
        (tmp_path / "AA000001.1.fasta").write_text(">cached\nACGT\n")
        mock_efetch = _make_mock_efetch({"AA000001.1": ">fresh\nACGT\n"})

        with patch("src.data.refseq_fetch.Entrez.efetch", side_effect=mock_efetch) as patched, \
             patch("src.data.refseq_fetch.time.sleep"):
            result = fetch_representative_sequences(
                accessions=["AA000001.1"],
                email="test@example.com",
                output_dir=tmp_path,
            )

        patched.assert_not_called()
        assert result.succeeded == ["AA000001.1"]
        assert (tmp_path / "AA000001.1.fasta").read_text() == ">cached\nACGT\n"

    def test_rate_limit_sleep_called_per_live_fetch(self, tmp_path: Path):
        fasta_map = {"AA000001.1": ">a\nACGT\n", "BB000002.1": ">b\nACGT\n"}
        with patch("src.data.refseq_fetch.Entrez.efetch", side_effect=_make_mock_efetch(fasta_map)), \
             patch("src.data.refseq_fetch.time.sleep") as mock_sleep:
            fetch_representative_sequences(
                accessions=list(fasta_map),
                email="test@example.com",
                output_dir=tmp_path,
                requests_per_second=5.0,
            )

        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(pytest.approx(0.2))

    def test_output_dir_created_if_missing(self, tmp_path: Path):
        nested = tmp_path / "nested" / "refseq"
        fasta_map = {"AA000001.1": ">a\nACGT\n"}
        with patch("src.data.refseq_fetch.Entrez.efetch", side_effect=_make_mock_efetch(fasta_map)), \
             patch("src.data.refseq_fetch.time.sleep"):
            fetch_representative_sequences(
                accessions=list(fasta_map),
                email="test@example.com",
                output_dir=nested,
            )

        assert nested.is_dir()
        assert (nested / "AA000001.1.fasta").exists()


# ---------------------------------------------------------------------------
# Unit tests: FetchResult.coverage
# ---------------------------------------------------------------------------


class TestFetchResultCoverage:
    def test_full_coverage(self):
        result = FetchResult(requested=2, succeeded=["A", "B"], failed={})
        assert result.coverage == 1.0

    def test_partial_coverage(self):
        result = FetchResult(requested=4, succeeded=["A"], failed={"B": "err"})
        assert result.coverage == 0.25

    def test_zero_requested_no_division_error(self):
        result = FetchResult(requested=0)
        assert result.coverage == 0.0
