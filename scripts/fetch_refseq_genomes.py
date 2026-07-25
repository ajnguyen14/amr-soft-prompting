"""RefSeq genome fetch entry point for the TA-proximity pipeline.

Downloads one genome assembly per species listed in
`data/manifests/refseq_species_accessions.json` (committed to the repo, so
every environment resolves the same species-to-assembly mapping) from
NCBI's Datasets API. The manifest itself is derived data (species names come
from the CARD/TADB organism overlap; accessions come from NCBI's
dataset_report endpoint) but is small and deterministic enough to check in,
so no environment has to re-run that resolution step -- only the (large,
gitignored) genome FASTA download below needs re-running per environment.

This mirrors preprocess_card.py's role: a single script any fresh clone can
run to reproduce the raw inputs `blast_runner.py` needs, rather than
`scp`-ing genome files between servers by hand.

Usage:
    python scripts/fetch_refseq_genomes.py --config configs/cpu_server.yaml
"""

import argparse
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config

# NCBI Datasets API v2 genome-by-accession download endpoint.
NCBI_DOWNLOAD_URL = (
    "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{}/download"
    "?include_annotation_type=GENOME_FASTA"
)

# Courtesy delay between requests; NCBI asks for <=3 req/sec without an API key.
REQUEST_DELAY_SECONDS = 0.5


def fetch_genomes(manifest_path: str, output_dir: str) -> dict[str, Any]:
    """Download the genome FASTA for every accession in the manifest.

    Skips accessions whose output file already exists and is non-empty, so
    the script is safe to re-run after a partial/interrupted download.

    Args:
        manifest_path: Path to data/manifests/refseq_species_accessions.json.
        output_dir: Directory to write one `{accession}.fna` file per genome,
            plus a `_download_log.json` summary.

    Returns:
        Summary dict with keys: total, ok, already_exists, failed, total_bytes.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log: list[dict[str, Any]] = []
    for i, entry in enumerate(manifest):
        accession = entry["accession"]
        dest = out_path / f"{accession}.fna"

        if dest.exists() and dest.stat().st_size > 0:
            log.append({"species": entry["species"], "accession": accession,
                        "status": "already_exists", "bytes": dest.stat().st_size})
            continue

        record = {"species": entry["species"], "accession": accession,
                  "status": None, "bytes": 0}
        try:
            req = urllib.request.Request(NCBI_DOWNLOAD_URL.format(accession))
            with urllib.request.urlopen(req, timeout=60) as resp:
                zip_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                fna_names = [n for n in zf.namelist() if n.endswith(".fna")]
                if not fna_names:
                    record["status"] = "no_fna_in_zip"
                else:
                    content = zf.read(fna_names[0])
                    dest.write_bytes(content)
                    record["status"] = "ok"
                    record["bytes"] = len(content)
        except Exception as e:
            record["status"] = f"error: {str(e)[:150]}"
        log.append(record)

        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(manifest)} done")
        time.sleep(REQUEST_DELAY_SECONDS)

    with open(out_path / "_download_log.json", "w") as f:
        json.dump(log, f, indent=2)

    ok = sum(1 for e in log if e["status"] in ("ok", "already_exists"))
    failed = [e for e in log if e["status"] not in ("ok", "already_exists")]
    total_bytes = sum(e["bytes"] for e in log)
    return {
        "total": len(manifest),
        "ok": ok,
        "failed": len(failed),
        "failed_entries": failed,
        "total_bytes": total_bytes,
    }


def main() -> None:
    """CLI entry point: python scripts/fetch_refseq_genomes.py --config <path>."""
    parser = argparse.ArgumentParser(
        description="Download RefSeq genome FASTAs listed in the committed species-accession manifest."
    )
    parser.add_argument(
        "--config", required=True, help="Path to a config YAML, e.g. configs/cpu_server.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    summary = fetch_genomes(
        config["paths"]["refseq_manifest"],
        config["paths"]["refseq_genomes_dir"],
    )

    print(f"Downloaded {summary['ok']}/{summary['total']} genomes "
          f"({summary['total_bytes'] / 1e9:.2f} GB)")
    if summary["failed"]:
        print(f"{summary['failed']} failed:")
        for e in summary["failed_entries"]:
            print(f"  {e['species']} ({e['accession']}): {e['status']}")


if __name__ == "__main__":
    main()
