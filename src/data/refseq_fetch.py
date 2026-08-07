"""Fetch RefSeq/GenBank nucleotide sequences for the TA-proximity pipeline's
representative accessions (CLAUDE.md TA-Proximity Pipeline Step 1).

Fetches are pinned to CARD's own recorded accession *version* (e.g.
'AA000001.1', not a bare 'AA000001') by passing the versioned accession
straight through as the Entrez `id` -- this keeps the fetched sequence
self-consistent with the exact genome build CARD's protein was drawn from,
rather than picking up the current live RefSeq version and risking
coordinate drift from reannotation (confirmed approach, see docs/STATUS.md).

Uses NCBI's E-utilities via Bio.Entrez, one accession per request, rate
limited client-side to NCBI's documented policy (3 req/sec without an API
key, 10 req/sec with one).
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from Bio import Entrez

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchResult:
    """Outcome of fetching a batch of representative accessions.

    Args:
        requested: Total number of accessions requested.
        succeeded: Accessions successfully fetched and written to disk.
        failed: Dict mapping accession to the error message for accessions
            that could not be fetched (e.g. withdrawn/suppressed records).
    """

    requested: int
    succeeded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Fraction of requested accessions successfully fetched, in [0, 1]."""
        return len(self.succeeded) / self.requested if self.requested else 0.0


def fetch_representative_sequences(
    accessions: list[str],
    email: str,
    output_dir: str | Path,
    api_key: Optional[str] = None,
    requests_per_second: float = 3.0,
) -> FetchResult:
    """Fetch nucleotide FASTA records for a list of versioned accessions.

    Each accession is written to `output_dir/<accession>.fasta`. Accessions
    already present on disk are skipped (not re-fetched) so an interrupted
    run can resume without re-downloading everything -- safe because
    fetches are pinned to an exact version, so a cached file can never be
    stale relative to what a re-fetch of the same accession would return.

    Args:
        accessions: Versioned DNA accessions to fetch, e.g. from
            refseq_representative.get_fetch_accession_list.
        email: Contact email for NCBI Entrez (required by NCBI's usage
            policy on every request).
        output_dir: Directory to write one FASTA file per accession into;
            created if it doesn't exist.
        api_key: Optional NCBI API key, raises the rate limit ceiling from
            3 to 10 req/sec.
        requests_per_second: Client-side rate limit. Must not exceed NCBI's
            policy limit (3 without a key, 10 with one) or requests risk
            being throttled/blocked.

    Returns:
        FetchResult summarizing which accessions succeeded and which
        failed (with the error message for each failure).
    """
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    delay_seconds = 1.0 / requests_per_second
    result = FetchResult(requested=len(accessions))

    for accession in accessions:
        dest = output_dir / f"{accession}.fasta"
        if dest.exists():
            result.succeeded.append(accession)
            continue

        try:
            with Entrez.efetch(
                db="nuccore", id=accession, rettype="fasta", retmode="text"
            ) as handle:
                fasta_text = handle.read()
        except Exception as exc:  # NCBI errors surface as generic urllib/IOError types
            logger.warning("Failed to fetch %s: %s", accession, exc)
            result.failed[accession] = str(exc)
            time.sleep(delay_seconds)
            continue

        # NCBI can return HTTP 200 with an empty body or an error page (not
        # an exception) for a withdrawn/suppressed accession -- a minimal
        # sanity check that this actually looks like FASTA, not a silent
        # write of garbage that would later poison build_blast_db.
        if not fasta_text.strip() or not fasta_text.lstrip().startswith(">"):
            logger.warning(
                "Fetched content for %s is not valid FASTA (empty or missing '>' header) "
                "-- treating as failed, not writing to disk",
                accession,
            )
            result.failed[accession] = "fetched content is not valid FASTA (empty or missing '>' header)"
            time.sleep(delay_seconds)
            continue

        # Write via a temp file + atomic rename (os.replace), not a direct
        # write_text -- a process killed mid-write (OOM, SSH drop) would
        # otherwise leave a truncated file at `dest` that the resume check
        # above (`dest.exists()`) would treat as permanently valid on every
        # future run, silently feeding a corrupt genome downstream.
        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        tmp_path.write_text(fasta_text)
        os.replace(tmp_path, dest)
        result.succeeded.append(accession)
        time.sleep(delay_seconds)

    logger.info(
        "Fetched %d/%d representative accessions (%.1f%% coverage)",
        len(result.succeeded),
        result.requested,
        result.coverage * 100,
    )
    return result
