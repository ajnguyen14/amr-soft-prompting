"""RefSeq fetch entry point for the TA-proximity pipeline (CLAUDE.md
TA-Proximity Pipeline Step 1).

Resolves CARD's 5,973 distinct DNA accessions down to one representative
accession per organism (src/data/refseq_representative.py), then fetches
each representative's nucleotide FASTA from NCBI via Bio.Entrez
(src/data/refseq_fetch.py), pinned to the exact version CARD recorded.

Usage:
    python scripts/fetch_refseq_representatives.py --config configs/ta_proximity_refseq.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.refseq_fetch import fetch_representative_sequences
from src.data.refseq_representative import (
    get_fetch_accession_list,
    load_aro_taxonomy_records,
    select_representative_accessions,
)
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch(config: dict[str, Any]) -> None:
    """Select representative accessions and fetch them per `config`.

    Args:
        config: Merged config dict from load_config, with 'paths' (card_json,
            refseq_output_dir) and 'refseq' (entrez_email, entrez_api_key,
            requests_per_second) sections.
    """
    records = load_aro_taxonomy_records(config["paths"]["card_json"])
    representatives = select_representative_accessions(records)
    accessions = get_fetch_accession_list(representatives)

    refseq_config = config["refseq"]
    result = fetch_representative_sequences(
        accessions=accessions,
        email=refseq_config["entrez_email"],
        output_dir=config["paths"]["refseq_output_dir"],
        api_key=refseq_config.get("entrez_api_key"),
        requests_per_second=refseq_config.get("requests_per_second", 3.0),
    )

    print(f"Requested {result.requested} representative accessions")
    print(f"Fetched   {len(result.succeeded)} ({result.coverage * 100:.1f}% coverage)")
    print(f"Failed    {len(result.failed)}")
    if result.failed:
        print("Failed accessions:")
        for accession, error in result.failed.items():
            print(f"  {accession}: {error}")


def main() -> None:
    """CLI entry point: python scripts/fetch_refseq_representatives.py --config <path>."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    fetch(config)


if __name__ == "__main__":
    main()
