"""BLAST coordinate-mapping entry point for the TA-proximity pipeline
(CLAUDE.md TA-Proximity Pipeline Step 1, second half).

Groups CARD protein sequences by their organism group's representative
RefSeq accession (src/data/refseq_representative.py) and BLASTs each group
against its already-fetched representative genome (src/data/refseq_fetch.py,
src/data/blast_runner.py) to place each ARO accession at a genomic
coordinate. Writes the resulting hits to a JSON artifact for the
not-yet-built Step 3 (same-replicon bp distance) to consume.

Usage:
    python scripts/run_blast_coordinate_mapping.py --config configs/ta_proximity_refseq.yaml
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.blast_runner import BlastHit, blast_card_against_representatives
from src.data.card_parser import load_card_dataset
from src.data.refseq_representative import (
    load_aro_taxonomy_records,
    map_aro_to_representative,
    select_representative_accessions,
)
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(config: dict[str, Any]) -> None:
    """Group CARD proteins by representative accession, BLAST, and save hits.

    Args:
        config: Merged config dict from load_config, with 'paths'
            (card_fasta, card_json, aro_index, refseq_output_dir,
            blastdb_dir, blast_hits_output, query_universe_output) and
            'blast' (bin_dir, min_identity, min_query_coverage, evalue)
            sections.
    """
    paths = config["paths"]

    taxonomy_records = load_aro_taxonomy_records(paths["card_json"])
    representatives = select_representative_accessions(taxonomy_records)
    mappings = map_aro_to_representative(taxonomy_records, representatives)
    mapping_by_aro = {m.aro_accession: m for m in mappings}

    card_records = load_card_dataset(paths["card_fasta"], paths["aro_index"], paths["card_json"])
    sequence_by_aro = {r.aro_accession: r.sequence for r in card_records}

    query_sequences_by_group: dict[str, dict[str, str]] = defaultdict(dict)
    unqueryable = 0
    for mapping in mappings:
        sequence = sequence_by_aro.get(mapping.aro_accession)
        if sequence is None:
            # In taxonomy_records (from card.json) but not in the training
            # protein FASTA -- not part of the dataset this project actually
            # classifies, so not worth BLASTing.
            unqueryable += 1
            continue
        query_sequences_by_group[mapping.representative_accession][mapping.aro_accession] = sequence

    total_queried = sum(len(v) for v in query_sequences_by_group.values())
    logger.info(
        "%d ARO accessions queryable (in CARD protein FASTA) across %d representative groups "
        "(%d taxonomy-resolved accessions have no FASTA sequence, skipped)",
        total_queried,
        len(query_sequences_by_group),
        unqueryable,
    )

    # Persist the exact query universe this step actually attempted (only
    # the `total_queried` queryable accessions, never the unqueryable ones)
    # as a shared artifact -- run_ta_proximity.py reads this instead of
    # recomputing taxonomy/representative/mapping logic a second time, which
    # previously risked (and in practice caused) 'unknown' silently
    # absorbing accessions this step never attempted to BLAST at all, on top
    # of genuine BLAST failures. Also carries used_own_accession per
    # accession so Step 3's output can be audited for the
    # substitution-genome bias (see AroRepresentativeMapping).
    query_universe = [
        {
            "aro_accession": aro_accession,
            "representative_accession": mapping_by_aro[aro_accession].representative_accession,
            "used_own_accession": mapping_by_aro[aro_accession].used_own_accession,
        }
        for group in query_sequences_by_group.values()
        for aro_accession in group
    ]
    query_universe_path = Path(paths["query_universe_output"])
    query_universe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(query_universe_path, "w", encoding="utf-8") as fh:
        json.dump(query_universe, fh, indent=2)

    blast_config = config["blast"]
    output_path = Path(paths["blast_hits_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rewritten after every representative group completes (not just once at
    # the very end) -- if a later group's BLAST call fails outright or the
    # process is killed, every already-completed group's hits are still on
    # disk rather than lost with the whole batch.
    accumulated_hits: list[BlastHit] = []

    def _checkpoint(current_hits: list[BlastHit]) -> None:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(hit) for hit in current_hits], fh, indent=2)

    def _on_group_complete(_representative_accession: str, group_hits: list[BlastHit]) -> None:
        accumulated_hits.extend(group_hits)
        _checkpoint(accumulated_hits)

    hits = blast_card_against_representatives(
        query_sequences_by_group,
        refseq_dir=paths["refseq_output_dir"],
        blastdb_dir=paths["blastdb_dir"],
        blast_bin_dir=blast_config.get("bin_dir"),
        min_identity=blast_config.get("min_identity", 95.0),
        min_query_coverage=blast_config.get("min_query_coverage", 90.0),
        evalue=blast_config.get("evalue", 1e-10),
        on_group_complete=_on_group_complete,
    )

    # Final write is the single source of truth for "the actual returned
    # hits" (equal to the last checkpoint on a clean run, but not dependent
    # on the checkpoint side effect having fired correctly).
    _checkpoint(hits)

    coverage = len(hits) / total_queried * 100 if total_queried else 0.0
    print(f"Queried    {total_queried} ARO accessions ({unqueryable} skipped, no FASTA sequence)")
    print(f"BLAST hits {len(hits)} ({coverage:.1f}% coverage)")
    print(f"Wrote hits to: {output_path}")
    print(f"Wrote query universe to: {query_universe_path}")


def main() -> None:
    """CLI entry point: python scripts/run_blast_coordinate_mapping.py --config <path>."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
