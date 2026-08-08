"""Parse CARD FASTA, ARO index, and card.json into CARDRecord objects."""

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from Bio import SeqIO

logger = logging.getLogger(__name__)

# Fallback category for any ARO accession absent from a loaded
# ta_proximity_results.json (e.g. the ~352 accessions with no CARD protein
# sequence to BLAST in the first place -- see docs/STATUS.md). Matches
# src/data/ta_proximity.py's `unknown` category: "did not map to RefSeq",
# a data-quality gap, never a real proximity signal.
_TA_PROXIMITY_UNKNOWN = "unknown"

# FASTA header format: >gb|<protein_acc>|ARO:<id>|<gene_name> [<organism>]
# Optional trailing qualifiers (e.g. " Partial") are captured and discarded.
_HEADER_RE = re.compile(
    r"^gb\|([^|]+)\|(ARO:\d+)\|([^\[]+?)(?:\s+\[([^\]]+)\].*)?$"
)


@dataclass
class CARDRecord:
    """A single CARD homolog model entry with sequence and V1 metadata.

    Args:
        aro_accession: ARO ontology accession (e.g. 'ARO:3002999').
        protein_accession: GenBank protein accession (e.g. 'ACT97415.1').
        gene_name: Short gene name from CARD (e.g. 'CblA-1').
        organism: Source organism string from the FASTA header.
        sequence: Amino acid sequence string.
        drug_classes: One or more drug classes this gene confers resistance to.
            Semicolon-delimited fields in aro_index.tsv are split into a list.
        resistance_mechanism: CARD resistance mechanism (e.g. 'antibiotic efflux').
        amr_gene_family: AMR gene family grouping (e.g. 'AAC(2\')').
        card_short_name: CARD short name identifier.
        ta_proximity_category: Run 3's soft-prompt conditioning value (V2
            TA-Proximity Pipeline Step 4) -- one of 'distance', 'no_ta_locus',
            or 'unknown', copied verbatim from
            src/data/ta_proximity.py's TAProximityResult.category (the V2
            decision, per Andreopoulos, collapsed fine-grained distance bins
            to this coarse 3-way categorical -- see CLAUDE.md's TA-Proximity
            Pipeline section). Empty string when load_card_dataset was called
            without ta_proximity_path (V1 and Run 1/2 callers, which never
            touch this field).
    """

    aro_accession: str
    protein_accession: str
    gene_name: str
    organism: str
    sequence: str
    drug_classes: list[str]
    resistance_mechanism: str
    amr_gene_family: str
    card_short_name: str
    ta_proximity_category: str = ""


def _parse_fasta_header(description: str) -> dict[str, str]:
    """Extract fields from a CARD FASTA description line.

    Args:
        description: Header string after the leading '>gb|' prefix, as returned
            by BioPython's SeqRecord.description.

    Returns:
        Dict with keys protein_accession, aro_accession, gene_name, organism.

    Raises:
        ValueError: If the header does not match the expected CARD format.
    """
    match = _HEADER_RE.match(description.strip())
    if not match:
        raise ValueError(f"Unexpected CARD FASTA header format: '{description}'")

    protein_acc, aro_acc, gene_name, organism = match.groups()
    return {
        "protein_accession": protein_acc.strip(),
        "aro_accession": aro_acc.strip(),
        "gene_name": gene_name.strip(),
        "organism": (organism or "").strip(),
    }


def parse_aro_index(tsv_path: str | Path) -> dict[str, dict[str, str]]:
    """Load aro_index.tsv into a dict keyed by ARO accession string.

    Public (not underscore-prefixed) because src/data/card_tadb_matcher.py
    needs raw row access (specifically the 'DNA Accession' column) --
    CARDRecord doesn't carry that field, so re-reading aro_index.tsv via
    this shared helper avoids duplicating the TSV-parsing logic. Note
    card_tadb_matcher.py itself is NOT part of the live TA-proximity
    pipeline (superseded -- see that module's docstring); this is a
    historical reason for making the function public, not a claim that its
    caller is currently load-bearing.

    Args:
        tsv_path: Path to aro_index.tsv.

    Returns:
        Dict mapping 'ARO:XXXXXXX' to the corresponding row dict.
    """
    index: dict[str, dict[str, str]] = {}
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            aro_acc = row["ARO Accession"].strip()
            index[aro_acc] = row
    logger.debug("Loaded %d entries from ARO index: %s", len(index), tsv_path)
    return index


def _load_ta_proximity_categories(ta_proximity_path: str | Path) -> dict[str, str]:
    """Load ARO accession -> TA-proximity category from ta_proximity_results.json.

    Args:
        ta_proximity_path: Path to the JSON list written by
            scripts/run_ta_proximity.py, e.g.
            data/processed/ta_proximity_results.json -- each entry a
            TAProximityResult dict (src/data/ta_proximity.py) with at least
            'aro_accession' and 'category' keys.

    Returns:
        Dict mapping ARO accession to its category string ('distance',
        'no_ta_locus', or 'unknown'). Accessions not present in the file
        (e.g. never queryable in Step 1) are simply absent from this dict --
        load_card_dataset falls back to _TA_PROXIMITY_UNKNOWN for those.
    """
    with open(ta_proximity_path, encoding="utf-8") as fh:
        results = json.load(fh)
    return {entry["aro_accession"]: entry["category"] for entry in results}


def load_card_dataset(
    fasta_path: str | Path,
    aro_index_path: str | Path,
    card_json_path: Optional[str | Path] = None,  # TODO: V2 — richer metadata from card.json
    ta_proximity_path: Optional[str | Path] = None,
) -> list[CARDRecord]:
    """Parse CARD FASTA and ARO index into a list of CARDRecord objects.

    Sequences are joined with ARO metadata on the ARO accession embedded in the
    FASTA header. Records whose ARO accession is absent from the index are
    skipped with a warning (should not occur in standard CARD releases).

    card.json is accepted for forward-compatibility but unused in V1 — Drug
    Class and Resistance Mechanism are sourced directly from aro_index.tsv.

    Args:
        fasta_path: Path to protein_fasta_protein_homolog_model.fasta.
        aro_index_path: Path to aro_index.tsv.
        card_json_path: Optional path to card.json (unused in V1).
        ta_proximity_path: Optional path to ta_proximity_results.json (V2
            Run 3 only -- see CLAUDE.md's TA-Proximity Pipeline). When given,
            each record's ta_proximity_category is joined in by ARO
            accession, defaulting to 'unknown' for any accession absent from
            the file. When omitted (V1, Run 1/2), every record's
            ta_proximity_category stays "" and get_label_vocabularies won't
            emit a 'ta_proximity' vocabulary.

    Returns:
        List of CARDRecord, one per FASTA sequence with metadata joined in.
    """
    aro_index = parse_aro_index(aro_index_path)
    ta_proximity_by_aro = (
        _load_ta_proximity_categories(ta_proximity_path) if ta_proximity_path else {}
    )

    records: list[CARDRecord] = []
    skipped = 0

    for seq_record in SeqIO.parse(str(fasta_path), "fasta"):
        try:
            header = _parse_fasta_header(seq_record.description)
        except ValueError as exc:
            logger.warning("Skipping malformed header: %s", exc)
            skipped += 1
            continue

        aro_acc = header["aro_accession"]
        if aro_acc not in aro_index:
            logger.warning("ARO accession %s not found in index — skipping", aro_acc)
            skipped += 1
            continue

        row = aro_index[aro_acc]

        # Drug Class is semicolon-delimited when a gene confers resistance to
        # multiple drug families (e.g. "macrolide antibiotic;lincosamide antibiotic").
        raw_drug_class = row.get("Drug Class", "").strip()
        drug_classes = [d.strip() for d in raw_drug_class.split(";") if d.strip()]

        # Only populated when a caller passed ta_proximity_path (V2 Run 3);
        # absent accessions default to 'unknown', same data-quality-gap
        # meaning as a genuine BLAST failure -- both mean "no genomic
        # coordinate", per CLAUDE.md's TA-Proximity Pipeline vocabulary.
        ta_proximity_category = (
            ta_proximity_by_aro.get(aro_acc, _TA_PROXIMITY_UNKNOWN) if ta_proximity_path else ""
        )

        records.append(
            CARDRecord(
                aro_accession=aro_acc,
                protein_accession=header["protein_accession"],
                gene_name=header["gene_name"],
                organism=header["organism"],
                sequence=str(seq_record.seq),
                drug_classes=drug_classes,
                resistance_mechanism=row.get("Resistance Mechanism", "").strip(),
                amr_gene_family=row.get("AMR Gene Family", "").strip(),
                card_short_name=row.get("CARD Short Name", "").strip(),
                ta_proximity_category=ta_proximity_category,
            )
        )

    logger.info(
        "Loaded %d CARD records (%d skipped) from %s",
        len(records),
        skipped,
        fasta_path,
    )
    return records


def get_label_vocabularies(records: list[CARDRecord]) -> dict[str, list[str]]:
    """Build sorted label vocabularies from a loaded CARD dataset.

    Used by dataset.py to construct integer label encodings. Each vocabulary
    entry is a sorted list of all unique values seen across the dataset.

    Args:
        records: List of CARDRecord from load_card_dataset.

    Returns:
        Dict with keys 'drug_class', 'resistance_mechanism', 'amr_gene_family',
        each mapping to a sorted list of unique label strings. Also includes
        'ta_proximity' (V2 Run 3's conditioning field) when at least one
        record was loaded with a non-empty ta_proximity_category, i.e. the
        caller passed load_card_dataset's ta_proximity_path -- otherwise the
        key is omitted entirely so Run 1/2 callers never see a spurious
        one-entry vocabulary.
    """
    drug_classes: set[str] = set()
    mechanisms: set[str] = set()
    families: set[str] = set()
    ta_proximity_categories: set[str] = set()

    for r in records:
        drug_classes.update(r.drug_classes)
        if r.resistance_mechanism:
            mechanisms.add(r.resistance_mechanism)
        if r.amr_gene_family:
            families.add(r.amr_gene_family)
        if r.ta_proximity_category:
            ta_proximity_categories.add(r.ta_proximity_category)

    vocabularies = {
        "drug_class": sorted(drug_classes),
        "resistance_mechanism": sorted(mechanisms),
        "amr_gene_family": sorted(families),
    }
    if ta_proximity_categories:
        vocabularies["ta_proximity"] = sorted(ta_proximity_categories)
    return vocabularies
