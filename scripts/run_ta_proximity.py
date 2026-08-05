"""TA-proximity distance categorization entry point (CLAUDE.md TA-Proximity
Pipeline Step 3).

Loads the BLAST hits produced by run_blast_coordinate_mapping.py and TADB 3.0
loci, then categorizes every CARD ARO accession Step 1 attempted to map into
'distance' (with a real same-replicon bp value), 'no_ta_locus', or 'unknown'.
Writes results to a JSON artifact and prints the category breakdown plus the
raw distance list -- the real distance histogram Step 4's bin edges must be
derived from (CLAUDE.md: bin edges are not chosen a priori).

This does NOT build the categorical nn.Embedding vocabulary itself (Step 4)
-- that still needs a human/design decision on bin edges once this script's
real histogram is in hand.

Usage:
    python scripts/run_ta_proximity.py --config configs/ta_proximity_refseq.yaml
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.blast_runner import BlastHit
from src.data.refseq_representative import (
    load_aro_taxonomy_records,
    map_aro_to_representative,
    select_representative_accessions,
)
from src.data.ta_proximity import compute_ta_proximity
from src.data.tadb_parser import load_all_tadb_loci
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_hits(blast_hits_path: str | Path) -> list[BlastHit]:
    """Load BlastHit records written by run_blast_coordinate_mapping.py.

    Args:
        blast_hits_path: Path to the JSON file written by
            run_blast_coordinate_mapping.py (list of BlastHit field dicts).

    Returns:
        Reconstructed BlastHit list.
    """
    with open(blast_hits_path, encoding="utf-8") as fh:
        raw_hits = json.load(fh)
    return [BlastHit(**raw) for raw in raw_hits]


def run(config: dict[str, Any]) -> None:
    """Categorize TA-locus proximity for every queryable ARO accession.

    Args:
        config: Merged config dict from load_config, with 'paths'
            (card_json, blast_hits_output, ta_proximity_output) and the raw
            TADB FASTA directory (data/raw, same as elsewhere in the
            pipeline -- TADBLocus loading takes a directory, not a file).
    """
    paths = config["paths"]

    hits = _load_hits(paths["blast_hits_output"])

    # TADB loci are loaded from the same data/raw directory tadb_parser.py
    # and card_tadb_matcher.py already use -- derived from card_json's
    # directory rather than a new config key, since all raw TADB/CARD files
    # live side by side there (see CLAUDE.md Datasets section).
    raw_dir = Path(paths["card_json"]).parent
    tadb_loci = load_all_tadb_loci(raw_dir)

    # The full query universe Step 1 attempted -- every ARO accession with
    # both a resolvable representative accession AND a CARD protein
    # sequence, matching exactly what run_blast_coordinate_mapping.py
    # attempted to BLAST (see that script's `unqueryable` accounting).
    taxonomy_records = load_aro_taxonomy_records(paths["card_json"])
    representatives = select_representative_accessions(taxonomy_records)
    mappings = map_aro_to_representative(taxonomy_records, representatives)
    all_aro_accessions = sorted({m.aro_accession for m in mappings})

    results = compute_ta_proximity(hits, tadb_loci, all_aro_accessions)

    output_path = Path(paths["ta_proximity_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)

    n_distance = sum(1 for r in results if r.category == "distance")
    n_no_ta = sum(1 for r in results if r.category == "no_ta_locus")
    n_unknown = sum(1 for r in results if r.category == "unknown")
    distances = sorted(r.distance_bp for r in results if r.category == "distance")

    print(f"Total ARO accessions: {len(results)}")
    print(f"  distance:    {n_distance}")
    print(f"  no_ta_locus: {n_no_ta}")
    print(f"  unknown:     {n_unknown}")
    print(f"Wrote results to: {output_path}")
    if distances:
        print(f"Distance histogram (bp), n={len(distances)}:")
        print(f"  min={distances[0]} max={distances[-1]}")
        for pct in (10, 25, 50, 75, 90):
            idx = min(len(distances) - 1, int(len(distances) * pct / 100))
            print(f"  p{pct}={distances[idx]}")
    else:
        print("No real distances computed -- nothing to histogram yet.")


def main() -> None:
    """CLI entry point: python scripts/run_ta_proximity.py --config <path>."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
