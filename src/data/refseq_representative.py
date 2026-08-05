"""Select one representative RefSeq/GenBank accession per organism for the
TA-proximity pipeline's RefSeq fetch step (CLAUDE.md TA-Proximity Pipeline
Step 1).

Fetching and BLASTing all ~5,973 distinct CARD DNA accessions individually is
impractical; grouping by organism (NCBI taxonomy ID) collapses this to ~740
representative genomes (confirmed against the real dataset). Within each
organism group, the most-common CARD DNA accession is picked as the
representative -- this maximizes how many ARO accessions in that group get
BLASTed against the exact genome build CARD originally recorded them
against (the project's confirmed version-pinning decision), rather than
substituting a different strain's assembly for every group member.

Grouping is keyed on NCBI_taxonomy_id, not the organism name string, because
CARD's own taxonomy tagging is inconsistent at the raw-accession level: four
DNA accessions in the real dataset (U00096.1, AL450380.1, AE004969.1,
MH423812.1) are each tagged with two different taxonomy IDs across different
ARO entries, so grouping by accession or by name string would either split
or merge organism groups incorrectly. Ties for most-common accession within
a group are broken by lexicographically smallest accession -- not sequence
length or an external RefSeq "representative genome" lookup -- so the whole
selection is deterministic from local data alone, with no network
dependency, consistent with the project's reproducibility requirements.
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AroTaxonomyRecord:
    """One CARD ARO accession's DNA accession and NCBI taxonomy, from card.json.

    Args:
        aro_accession: CARD ARO accession (e.g. 'ARO:3002999').
        dna_accession: CARD's recorded, versioned DNA Accession (e.g. 'GU256745.1').
        taxonomy_id: NCBI taxonomy ID for the source organism (e.g. '511145').
        taxonomy_name: NCBI taxonomy name (e.g. 'Escherichia coli str. K-12 substr. MG1655').
    """

    aro_accession: str
    dna_accession: str
    taxonomy_id: str
    taxonomy_name: str


def load_aro_taxonomy_records(card_json_path: str | Path) -> list[AroTaxonomyRecord]:
    """Extract (ARO accession, DNA accession, taxonomy) triples from card.json.

    card.json's per-entry ARO_accession field is unprefixed (e.g. '3002999');
    this normalizes it to the 'ARO:3002999' form used everywhere else in the
    codebase (aro_index.tsv, CARDRecord, AccessionMatch).

    Args:
        card_json_path: Path to card.json.

    Returns:
        One AroTaxonomyRecord per card.json entry that has both a DNA
        accession and NCBI taxonomy info. Entries missing either (e.g. no
        model_sequences block, or an incomplete taxonomy block) are skipped
        and counted in the log line.
    """
    with open(card_json_path, encoding="utf-8") as fh:
        card = json.load(fh)

    records: list[AroTaxonomyRecord] = []
    skipped = 0
    for entry in card.values():
        if not isinstance(entry, dict) or "model_sequences" not in entry:
            continue

        aro_acc = entry.get("ARO_accession")
        if not aro_acc:
            skipped += 1
            continue
        if not aro_acc.startswith("ARO:"):
            aro_acc = f"ARO:{aro_acc}"

        for seq in entry.get("model_sequences", {}).get("sequence", {}).values():
            dna_acc = seq.get("dna_sequence", {}).get("accession")
            taxonomy = seq.get("NCBI_taxonomy", {})
            tax_id = taxonomy.get("NCBI_taxonomy_id")
            tax_name = taxonomy.get("NCBI_taxonomy_name")
            if not (dna_acc and tax_id and tax_name):
                skipped += 1
                continue
            records.append(
                AroTaxonomyRecord(
                    aro_accession=aro_acc,
                    dna_accession=dna_acc,
                    taxonomy_id=tax_id,
                    taxonomy_name=tax_name,
                )
            )

    logger.info(
        "Loaded %d ARO/DNA-accession/taxonomy records from %d card.json entries (%d skipped)",
        len(records),
        len(card),
        skipped,
    )
    return records


@dataclass(frozen=True)
class RepresentativeAccession:
    """The chosen representative DNA accession for one organism (taxonomy) group.

    Args:
        taxonomy_id: NCBI taxonomy ID identifying the organism group.
        taxonomy_name: A taxonomy name seen for this group (the first
            encountered -- organisms with a single taxonomy_id may still
            have minor name-string variants across entries).
        representative_accession: The DNA accession chosen to represent this
            group -- the most-common accession among the group's ARO
            entries, tie-broken lexicographically.
        group_size: Number of ARO accessions in this taxonomy group.
        candidate_accession_count: Number of distinct DNA accessions seen in
            this group (1 means no choice was actually needed).
        tie_broken: True if more than one candidate accession tied for
            most-common and the lexicographic tiebreak was used.
    """

    taxonomy_id: str
    taxonomy_name: str
    representative_accession: str
    group_size: int
    candidate_accession_count: int
    tie_broken: bool


def select_representative_accessions(
    records: list[AroTaxonomyRecord],
) -> list[RepresentativeAccession]:
    """Pick one representative DNA accession per taxonomy (organism) group.

    Within each group, the most-common DNA accession among the group's ARO
    entries is chosen -- this keeps the largest possible share of that
    group's entries BLASTed against the exact genome CARD recorded them
    against. Ties are broken by lexicographically smallest accession, kept
    deterministic and free of any external RefSeq lookup.

    Args:
        records: AroTaxonomyRecord list, e.g. from load_aro_taxonomy_records.

    Returns:
        One RepresentativeAccession per distinct taxonomy_id in records.
    """
    groups: dict[str, list[AroTaxonomyRecord]] = defaultdict(list)
    for rec in records:
        groups[rec.taxonomy_id].append(rec)

    representatives: list[RepresentativeAccession] = []
    tie_broken_count = 0
    for taxonomy_id, group_records in groups.items():
        accession_counts = Counter(r.dna_accession for r in group_records)
        top_count = max(accession_counts.values())
        tied_candidates = sorted(
            acc for acc, count in accession_counts.items() if count == top_count
        )
        representative = tied_candidates[0]
        tie_broken = len(tied_candidates) > 1
        if tie_broken:
            tie_broken_count += 1

        representatives.append(
            RepresentativeAccession(
                taxonomy_id=taxonomy_id,
                taxonomy_name=group_records[0].taxonomy_name,
                representative_accession=representative,
                group_size=len(group_records),
                candidate_accession_count=len(accession_counts),
                tie_broken=tie_broken,
            )
        )

    logger.info(
        "Selected %d representative accessions from %d taxonomy groups (%d tie-broken)",
        len(representatives),
        len(groups),
        tie_broken_count,
    )
    return representatives


@dataclass(frozen=True)
class AroRepresentativeMapping:
    """Maps one ARO accession to the representative accession for its organism group.

    Args:
        aro_accession: CARD ARO accession.
        own_dna_accession: CARD's own recorded DNA accession for this ARO entry.
        taxonomy_id: NCBI taxonomy ID for this ARO entry's organism group.
        representative_accession: The group's chosen representative accession
            (see select_representative_accessions).
        used_own_accession: True if own_dna_accession IS the group's
            representative -- i.e. this entry needs no strain substitution.
    """

    aro_accession: str
    own_dna_accession: str
    taxonomy_id: str
    representative_accession: str
    used_own_accession: bool


def map_aro_to_representative(
    records: list[AroTaxonomyRecord],
    representatives: list[RepresentativeAccession],
) -> list[AroRepresentativeMapping]:
    """Join each ARO record to its organism group's representative accession.

    Reports, per ARO accession, whether it will be BLASTed against its own
    originally-recorded accession or a substituted one from the same
    organism group -- the coverage/accuracy tradeoff CLAUDE.md's TA-Proximity
    Pipeline Step 1 asks to be recorded as a reportable number, not just a
    pipeline detail.

    Args:
        records: AroTaxonomyRecord list, e.g. from load_aro_taxonomy_records.
        representatives: RepresentativeAccession list, e.g. from
            select_representative_accessions.

    Returns:
        One AroRepresentativeMapping per input record.
    """
    rep_by_taxonomy = {r.taxonomy_id: r.representative_accession for r in representatives}

    mappings = [
        AroRepresentativeMapping(
            aro_accession=rec.aro_accession,
            own_dna_accession=rec.dna_accession,
            taxonomy_id=rec.taxonomy_id,
            representative_accession=rep_by_taxonomy[rec.taxonomy_id],
            used_own_accession=rec.dna_accession == rep_by_taxonomy[rec.taxonomy_id],
        )
        for rec in records
    ]

    substituted = sum(1 for m in mappings if not m.used_own_accession)
    logger.info(
        "%d/%d ARO accessions mapped to a substituted (non-own) representative accession",
        substituted,
        len(mappings),
    )
    return mappings


def get_fetch_accession_list(representatives: list[RepresentativeAccession]) -> list[str]:
    """Return the sorted, deduplicated accession list for the RefSeq fetch step.

    Args:
        representatives: RepresentativeAccession list, e.g. from
            select_representative_accessions.

    Returns:
        Sorted list of distinct representative DNA accessions to fetch.
    """
    return sorted({r.representative_accession for r in representatives})
