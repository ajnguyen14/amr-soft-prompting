"""TA-locus proximity preprocessing entry point (V2, Run 3's conditioning input).

Runs the full TA-proximity pipeline from CLAUDE.md's "TA-Proximity Pipeline"
section in two phases:

  Phase 1 (Steps 1-2, cached): BLAST-map CARD protein sequences onto RefSeq
  genomic coordinates, and parse TADB 3.0's FASTA headers into TA-locus
  coordinates. This is the expensive step (tblastn over thousands of CARD
  queries against a multi-genome database) and doesn't depend on any binning
  decision, so its output is cached to disk and reused by default across
  runs -- pass --rebuild-mapping to force it to rerun (e.g. after changing
  BLAST thresholds or updating input data).

  Phase 2 (Steps 3-4, always runs): compute same-replicon bp distance and
  bucket it into the categorical vocabulary ta_proximity.py defines. Cheap
  interval arithmetic, safe to rerun on every invocation. Requires
  config['ta_proximity']['distance_bin_edges_bp'] to be set -- on a fresh
  project this won't exist yet, so the first run prints a distance summary
  and stops with instructions, rather than guessing bin edges itself
  (CLAUDE.md: edges must come from the real histogram, not be chosen a
  priori).

Usage:
    python scripts/preprocess_ta_proximity.py --config configs/cpu_server.yaml
    python scripts/preprocess_ta_proximity.py --config configs/cpu_server.yaml --rebuild-mapping
"""

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Make the project root importable so `from src...` works when this script is
# run directly, mirroring preprocess_card.py.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.blast_runner import BlastMapping, build_blast_database, map_card_to_refseq
from src.data.card_parser import load_card_dataset
from src.data.tadb_parser import TALocus, parse_tadb_fasta
from src.data.ta_proximity import (
    ProximityResult,
    build_category_vocab,
    categorize,
    compute_same_replicon_distances,
)
from src.utils.config import load_config

MAPPING_CACHE_FILENAME = "ta_proximity_mapping_cache.pkl"
PROXIMITY_ARTIFACT_FILENAME = "ta_proximity_artifact.pkl"

# Config path keys -> TADB 'source' label, per CLAUDE.md: use both files for
# usable coverage (403 high-confidence 'exp' pairs, larger lower-confidence
# 'pre' set). Regulator files are intentionally excluded -- see CLAUDE.md's
# TADB 3.0 dataset notes.
_TADB_PATH_KEYS = (
    ("tadb_toxin_exp", "exp"),
    ("tadb_antitoxin_exp", "exp"),
    ("tadb_toxin_pre", "pre"),
    ("tadb_antitoxin_pre", "pre"),
)


@dataclass
class MappingCache:
    """Cached Step 1-2 output: CARD-to-RefSeq mappings and parsed TADB loci.

    Args:
        blast_mappings: aro_accession -> BlastMapping, from
            blast_runner.map_card_to_refseq.
        blast_coverage_pct: Percent of CARD accessions successfully mapped
            to RefSeq (CLAUDE.md's reportable BLAST coverage number).
        ta_loci: Combined TALocus list, all four TADB files.
        blast_thresholds: The config['blast'] dict this cache was produced
            under. Compared against the current config on reuse so a
            threshold change (e.g. min_pident) can't be silently ignored.
    """

    blast_mappings: dict[str, BlastMapping]
    blast_coverage_pct: float
    ta_loci: list[TALocus]
    blast_thresholds: dict[str, float]


def _ensure_blast_database(paths: dict[str, str]) -> None:
    """Build the RefSeq BLAST database if it doesn't already exist.

    makeblastdb -parse_seqids always writes a '.nin' index file; its
    presence is used as the build-complete marker, matching
    fetch_refseq_genomes.py's existing skip-if-present convention elsewhere
    in this pipeline.

    Args:
        paths: config['paths'], must contain 'refseq_genomes_dir' and
            'refseq_blast_db'.
    """
    db_marker = Path(f"{paths['refseq_blast_db']}.nin")
    if db_marker.exists():
        return
    build_blast_database(paths["refseq_genomes_dir"], paths["refseq_blast_db"])


def _run_phase1(config: dict[str, Any]) -> MappingCache:
    """Run Steps 1-2: BLAST-map CARD onto RefSeq, parse TADB 3.0.

    Args:
        config: Merged config dict with a 'paths' section (card_fasta,
            aro_index, card_json, refseq_genomes_dir, refseq_blast_db, and
            the four tadb_* keys in _TADB_PATH_KEYS) and a 'blast' section
            (min_pident, min_qcov, max_evalue).

    Returns:
        A MappingCache ready to be cached to disk and consumed by phase 2.
    """
    paths = config["paths"]
    blast_config = config["blast"]
    # Only the thresholds that actually affect which hits pass are compared
    # for cache staleness -- performance knobs like num_threads live in the
    # same 'blast:' config section but must never trigger a false "rebuild
    # required" just because someone tuned them between runs.
    blast_thresholds = {
        "min_pident": blast_config["min_pident"],
        "min_qcov": blast_config["min_qcov"],
        "max_evalue": blast_config["max_evalue"],
    }

    records = load_card_dataset(paths["card_fasta"], paths["aro_index"], paths.get("card_json"))

    _ensure_blast_database(paths)
    blast_mappings, coverage_pct = map_card_to_refseq(
        records,
        paths["refseq_blast_db"],
        min_pident=blast_thresholds["min_pident"],
        min_qcov=blast_thresholds["min_qcov"],
        max_evalue=blast_thresholds["max_evalue"],
        num_threads=blast_config.get("num_threads", 1),
    )

    ta_loci: list[TALocus] = []
    for path_key, source in _TADB_PATH_KEYS:
        ta_loci.extend(parse_tadb_fasta(paths[path_key], source=source))

    return MappingCache(
        blast_mappings=blast_mappings,
        blast_coverage_pct=coverage_pct,
        ta_loci=ta_loci,
        blast_thresholds=dict(blast_thresholds),
    )


def _save_mapping_cache(cache: MappingCache, output_dir: str | Path) -> Path:
    """Pickle a MappingCache to output_dir/MAPPING_CACHE_FILENAME."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / MAPPING_CACHE_FILENAME
    with open(cache_path, "wb") as fh:
        pickle.dump(cache, fh)
    return cache_path


def _load_mapping_cache(output_dir: str | Path) -> Optional[MappingCache]:
    """Load a cached MappingCache from output_dir, or None if absent."""
    cache_path = Path(output_dir) / MAPPING_CACHE_FILENAME
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as fh:
        return pickle.load(fh)


def _get_mapping_cache(config: dict[str, Any], rebuild_mapping: bool) -> MappingCache:
    """Reuse a cached phase-1 MappingCache if valid, else (re)compute it.

    Reuse is the default (rebuild_mapping=False): if a cache exists on disk
    and was built under the same BLAST thresholds the current config
    specifies, it's returned as-is without rerunning BLAST or re-parsing
    TADB.

    Args:
        config: Merged config dict, see _run_phase1.
        rebuild_mapping: If True, always recompute phase 1 and overwrite the
            cache, ignoring any existing cache on disk.

    Returns:
        A MappingCache, either freshly computed or loaded from disk.

    Raises:
        ValueError: If a cached MappingCache exists but was produced under
            different BLAST thresholds than the current config's 'blast'
            section. Reusing it silently would mean phase 2 conditions on
            mappings computed under thresholds nobody currently believes in
            -- pass --rebuild-mapping to regenerate it instead.
    """
    output_dir = config["paths"]["output_dir"]

    if not rebuild_mapping:
        cached = _load_mapping_cache(output_dir)
        if cached is not None:
            current_thresholds = {
                "min_pident": config["blast"]["min_pident"],
                "min_qcov": config["blast"]["min_qcov"],
                "max_evalue": config["blast"]["max_evalue"],
            }
            if cached.blast_thresholds != current_thresholds:
                raise ValueError(
                    f"Cached phase-1 mapping at {Path(output_dir) / MAPPING_CACHE_FILENAME} "
                    f"was built with blast thresholds {cached.blast_thresholds}, but the "
                    f"current config specifies {current_thresholds}. Re-run with "
                    "--rebuild-mapping to regenerate it under the current thresholds."
                )
            print(
                f"Reusing cached phase-1 mapping ({len(cached.blast_mappings)} accessions, "
                f"{len(cached.ta_loci)} TA loci)."
            )
            return cached

    cache = _run_phase1(config)
    cache_path = _save_mapping_cache(cache, output_dir)
    print(f"Wrote phase-1 mapping cache to: {cache_path}")
    return cache


def _print_distance_histogram(results: dict[str, ProximityResult]) -> None:
    """Print a min/percentile/max summary of real same-replicon distances.

    This is the mechanism for choosing config['ta_proximity']['distance_bin_edges_bp']
    per CLAUDE.md's "set from the actual distance histogram" requirement --
    always printed, regardless of whether bin edges are configured yet.

    Args:
        results: aro_accession -> ProximityResult, from
            compute_same_replicon_distances.
    """
    distances = sorted(r.distance_bp for r in results.values() if r.distance_bp is not None)
    unmapped = sum(1 for r in results.values() if not r.mapped)
    no_locus = sum(1 for r in results.values() if r.mapped and r.distance_bp is None)

    print(
        f"Same-replicon distance summary: {len(distances)} accessions with a real distance, "
        f"{unmapped} unmapped, {no_locus} mapped with no TA locus on their replicon."
    )
    if distances:
        def _percentile(p: float) -> int:
            return distances[min(int(p * len(distances)), len(distances) - 1)]

        print(
            f"  min={distances[0]}bp  p25={_percentile(0.25)}bp  p50={_percentile(0.50)}bp  "
            f"p75={_percentile(0.75)}bp  p90={_percentile(0.90)}bp  max={distances[-1]}bp"
        )


def _save_proximity_artifact(
    category_by_accession: dict[str, str],
    category_vocab: list[str],
    results: dict[str, ProximityResult],
    bin_edges: list[int],
    output_dir: str | Path,
) -> Path:
    """Pickle the final phase-2 artifact to output_dir/PROXIMITY_ARTIFACT_FILENAME.

    Args:
        category_by_accession: aro_accession -> category string, from categorize().
        category_vocab: Ordered vocabulary, from build_category_vocab() --
            list index is the nn.Embedding row for that category.
        results: aro_accession -> ProximityResult, from
            compute_same_replicon_distances -- raw distances are kept
            alongside the binned categories for diagnostics/reproducibility.
        bin_edges: The bp thresholds used, stamped in for reproducibility.
        output_dir: Directory to write the artifact into; created if missing.

    Returns:
        Path to the written artifact file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / PROXIMITY_ARTIFACT_FILENAME

    with open(artifact_path, "wb") as fh:
        pickle.dump(
            {
                "category_by_accession": category_by_accession,
                "category_vocab": category_vocab,
                "distance_by_accession": {a: r.distance_bp for a, r in results.items()},
                "bin_edges_bp": bin_edges,
            },
            fh,
        )
    return artifact_path


def _run_phase2(cache: MappingCache, config: dict[str, Any]) -> Path:
    """Run Steps 3-4: same-replicon distance + categorical binning.

    Args:
        cache: MappingCache from phase 1 (freshly computed or reused).
        config: Merged config dict; reads
            config['ta_proximity']['distance_bin_edges_bp'].

    Returns:
        Path to the written final artifact.

    Raises:
        ValueError: If distance_bin_edges_bp isn't set in config yet --
            expected on a project's first run, before the distance histogram
            printed here has been inspected to choose them.
    """
    results = compute_same_replicon_distances(cache.blast_mappings, cache.ta_loci)
    _print_distance_histogram(results)

    bin_edges = config.get("ta_proximity", {}).get("distance_bin_edges_bp")
    if not bin_edges:
        raise ValueError(
            "config['ta_proximity']['distance_bin_edges_bp'] is not set. Inspect the distance "
            "summary printed above, choose bin edges, add them under a 'ta_proximity:' section "
            "in config, then re-run this script (phase 1's cache will be reused automatically)."
        )

    category_vocab = build_category_vocab(bin_edges)
    category_by_accession = categorize(results, bin_edges)

    artifact_path = _save_proximity_artifact(
        category_by_accession, category_vocab, results, bin_edges, config["paths"]["output_dir"]
    )
    _print_summary(cache, category_by_accession, category_vocab, artifact_path)
    return artifact_path


def _print_summary(
    cache: MappingCache,
    category_by_accession: dict[str, str],
    category_vocab: list[str],
    artifact_path: Path,
) -> None:
    """Print BLAST coverage and the final category distribution to stdout."""
    print(f"Wrote TA-proximity artifact to: {artifact_path}")
    print(f"BLAST coverage: {cache.blast_coverage_pct:.2f}% of CARD accessions mapped to RefSeq")
    print("Category distribution:")
    counts = {category: 0 for category in category_vocab}
    for category in category_by_accession.values():
        counts[category] += 1
    for category in category_vocab:
        print(f"  {category}: {counts[category]}")


def preprocess(config: dict[str, Any], rebuild_mapping: bool) -> Path:
    """Run the full TA-proximity pipeline (phase 1 + phase 2) per `config`.

    Args:
        config: Merged config dict from load_config.
        rebuild_mapping: If True, force-recompute phase 1 even if a valid
            cache exists.

    Returns:
        Path to the written final TA-proximity artifact.
    """
    cache = _get_mapping_cache(config, rebuild_mapping)
    return _run_phase2(cache, config)


def main() -> None:
    """CLI entry point: python scripts/preprocess_ta_proximity.py --config <path> [--rebuild-mapping]."""
    parser = argparse.ArgumentParser(
        description="Compute TA-locus proximity categories for CARD ARO accessions."
    )
    parser.add_argument(
        "--config", required=True, help="Path to a config YAML, e.g. configs/cpu_server.yaml"
    )
    parser.add_argument(
        "--rebuild-mapping",
        action="store_true",
        help=(
            "Force-rerun the BLAST mapping and TADB parsing (Steps 1-2) even if a cached "
            "result exists -- e.g. after changing BLAST thresholds or updating input data. "
            "Reuses the cache by default."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess(config, rebuild_mapping=args.rebuild_mapping)


if __name__ == "__main__":
    main()
