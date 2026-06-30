"""PyTorch Dataset wrapping CARDRecord objects from card_parser.py."""

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
