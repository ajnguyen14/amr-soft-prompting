"""PyTorch Dataset wrapping CARDRecord objects from card_parser.py."""

import random

import torch
from torch.utils.data import Dataset

from src.data.card_parser import CARDRecord


class AMRDataset(Dataset):
    """PyTorch Dataset for AMR gene sequences with CARD metadata labels.

    Multi-label encoding rationale
    --------------------------------
    57% of CARD records (3463 / 6052) carry more than one drug_class label
    (distribution: 2589 × 1 class, 1482 × 2, 1593 × 3, 308 × 4, long tail to 14).
    Multi-label is the common case, not an edge case, so drug_class_labels is a
    fixed-length multi-hot float tensor sized to the full drug_class vocabulary.
    This matches the input shape required by BCEWithLogitsLoss and avoids the need
    for padding or a custom collate_fn.

    resistance_mechanism and amr_gene_family are genuinely single-label — each
    CARD entry has exactly one mechanism and one gene family — so they are encoded
    as plain integer indices for use with CrossEntropyLoss downstream.

    Collation note
    --------------
    PyTorch's default DataLoader collation works without a custom collate_fn:
      - "sequence" and "aro_accession" (str) are gathered into lists.
      - "drug_class_labels" (same-shape float tensors) are stacked to (B, |vocab|).
      - "resistance_mechanism" and "amr_gene_family" (scalar long tensors) are
        stacked to (B,).
    Sequences remain variable-length strings; tokenization and padding are deferred
    to the ESM-2 wrapper, which owns the ESM tokenizer.

    Args:
        records: List of CARDRecord from card_parser.load_card_dataset.
        label_vocabularies: Dict returned by card_parser.get_label_vocabularies,
            with keys 'drug_class', 'resistance_mechanism', 'amr_gene_family',
            each mapping to a sorted list of unique label strings.
    """

    def __init__(
        self,
        records: list[CARDRecord],
        label_vocabularies: dict[str, list[str]],
    ) -> None:
        self._records = records

        self._drug_class_vocab: list[str] = label_vocabularies["drug_class"]
        self._drug_class_to_idx: dict[str, int] = {
            label: idx for idx, label in enumerate(self._drug_class_vocab)
        }
        self._mechanism_to_idx: dict[str, int] = {
            label: idx for idx, label in enumerate(label_vocabularies["resistance_mechanism"])
        }
        self._family_to_idx: dict[str, int] = {
            label: idx for idx, label in enumerate(label_vocabularies["amr_gene_family"])
        }

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """Return a single sample as a dict.

        Args:
            idx: Integer index into the record list.

        Returns:
            Dict with keys:
                sequence (str): Raw amino acid sequence. Tokenization is deferred
                    to the ESM-2 wrapper.
                drug_class_labels (torch.FloatTensor): Multi-hot vector of shape
                    (|drug_class_vocab|,). Built here because BCEWithLogitsLoss
                    requires this fixed shape; there is no benefit to deferring it
                    to collation or loss time.
                resistance_mechanism (torch.LongTensor): Scalar integer index into
                    the resistance_mechanism vocabulary.
                amr_gene_family (torch.LongTensor): Scalar integer index into the
                    amr_gene_family vocabulary.
                aro_accession (str): ARO accession string, passed through unchanged
                    for traceability and debugging.
        """
        record = self._records[idx]

        # 57% of CARD records have >1 drug class — multi-hot is the correct encoding.
        # Unknown drug classes (e.g. from a held-out split) are silently skipped
        # so the vector stays all-zeros for that class rather than raising.
        drug_class_vec = torch.zeros(len(self._drug_class_vocab), dtype=torch.float32)
        for dc in record.drug_classes:
            if dc in self._drug_class_to_idx:
                drug_class_vec[self._drug_class_to_idx[dc]] = 1.0

        return {
            "sequence": record.sequence,
            "drug_class_labels": drug_class_vec,
            "resistance_mechanism": torch.tensor(
                self._mechanism_to_idx[record.resistance_mechanism], dtype=torch.long
            ),
            "amr_gene_family": torch.tensor(
                self._family_to_idx[record.amr_gene_family], dtype=torch.long
            ),
            "aro_accession": record.aro_accession,
        }


def split_dataset(
    records: list[CARDRecord],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, list[CARDRecord]]:
    """Split CARD records into train/val/test sets, split on ARO accession.

    The split unit is the ARO accession, not the individual sequence: CARD can
    have multiple protein sequences under one ARO accession, and letting
    sequence variants of the same accession land in different splits would leak
    near-duplicates of a test-set gene into training. Grouping by accession
    first guarantees every sequence for a given accession lands in the same
    split.

    Stratified by resistance_mechanism so rare mechanisms aren't concentrated
    into a single split. Deterministic given a fixed seed: shuffling uses a
    local random.Random instance, not the global random module, so calling
    this function has no side effect on unrelated code's RNG state.

    Args:
        records: List of CARDRecord from card_parser.load_card_dataset.
        train_frac: Fraction of ARO accessions assigned to the train split.
        val_frac: Fraction of ARO accessions assigned to the val split.
        test_frac: Fraction of ARO accessions assigned to the test split.
        seed: Seed for the local RNG controlling the per-mechanism shuffle.
            Project default is 42.

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to the list of
        CARDRecord whose aro_accession was assigned to that split. Every
        input record appears in exactly one output list.

    Raises:
        ValueError: If train_frac + val_frac + test_frac does not sum to 1.0
            (within floating-point tolerance), or if two records share an
            aro_accession but disagree on resistance_mechanism.
    """
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 1e-9:
        raise ValueError(
            f"train_frac + val_frac + test_frac must sum to 1.0, got "
            f"{train_frac} + {val_frac} + {test_frac} = {total_frac}"
        )

    # Group records by ARO accession (the split unit) and resolve each
    # accession's resistance_mechanism (the stratification key).
    records_by_accession: dict[str, list[CARDRecord]] = {}
    mechanism_by_accession: dict[str, str] = {}
    for record in records:
        records_by_accession.setdefault(record.aro_accession, []).append(record)
        existing_mechanism = mechanism_by_accession.get(record.aro_accession)
        if existing_mechanism is None:
            mechanism_by_accession[record.aro_accession] = record.resistance_mechanism
        elif existing_mechanism != record.resistance_mechanism:
            raise ValueError(
                f"ARO accession {record.aro_accession} has inconsistent "
                f"resistance_mechanism values: '{existing_mechanism}' vs "
                f"'{record.resistance_mechanism}'"
            )

    # Group accessions by mechanism so each mechanism is split 80/10/10
    # independently, then shuffle each group deterministically.
    accessions_by_mechanism: dict[str, list[str]] = {}
    for accession, mechanism in mechanism_by_accession.items():
        accessions_by_mechanism.setdefault(mechanism, []).append(accession)

    rng = random.Random(seed)
    split_accessions: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for mechanism in sorted(accessions_by_mechanism):
        # Sorted before shuffling so the result depends only on `seed`, not on
        # the incidental order records appeared in the input list.
        accessions = sorted(accessions_by_mechanism[mechanism])
        rng.shuffle(accessions)

        n = len(accessions)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        # Remainder (not int(n * test_frac)) guarantees the three counts sum to
        # n exactly regardless of rounding. A mechanism with very few
        # accessions may end up with zero in train and/or val as a result --
        # an inherent limit of stratifying rare classes across three splits.
        split_accessions["train"].extend(accessions[:n_train])
        split_accessions["val"].extend(accessions[n_train : n_train + n_val])
        split_accessions["test"].extend(accessions[n_train + n_val :])

    accession_to_split = {
        accession: split_name
        for split_name, accessions in split_accessions.items()
        for accession in accessions
    }

    result: dict[str, list[CARDRecord]] = {"train": [], "val": [], "test": []}
    for accession, accession_records in records_by_accession.items():
        result[accession_to_split[accession]].extend(accession_records)

    return result
