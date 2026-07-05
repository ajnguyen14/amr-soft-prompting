"""MLP classification head: shared trunk with three task-specific output heads."""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Shared-trunk MLP producing logits for three AMR classification tasks.

    A single hidden layer (trunk) feeds three independent linear heads, one per
    CARD metadata task. drug_class is multi-label (paired with BCEWithLogitsLoss
    downstream); resistance_mechanism and amr_gene_family are single-label (paired
    with CrossEntropyLoss downstream). Equal loss weighting across the three tasks
    was settled during the 2026-07-02 design session (see docs/STATUS.md).

    Args:
        input_dim: Width of the incoming representation from ESM2Wrapper. Mode
            dependent — embed_dim for 'internal' injection, or
            embed_dim + N * embed_dim for 'external' injection. Passed in by the
            caller rather than inferred here.
        hidden_dim: Width of the shared trunk's hidden layer. From config.
        dropout: Dropout probability applied after the trunk's ReLU. From config.
        num_drug_classes: Size of the drug_class vocabulary, from
            card_parser.get_label_vocabularies().
        num_mechanisms: Size of the resistance_mechanism vocabulary, from
            card_parser.get_label_vocabularies().
        num_families: Size of the amr_gene_family vocabulary, from
            card_parser.get_label_vocabularies().
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_drug_classes: int,
        num_mechanisms: int,
        num_families: int,
    ) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.drug_class_head = nn.Linear(hidden_dim, num_drug_classes)
        self.resistance_mechanism_head = nn.Linear(hidden_dim, num_mechanisms)
        self.amr_gene_family_head = nn.Linear(hidden_dim, num_families)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the shared trunk then all three task heads.

        Args:
            x: Tensor of shape (B, input_dim) — pooled representation from
                ESM2Wrapper (internal or external injection mode).

        Returns:
            Dict with keys:
                drug_class: (B, num_drug_classes) logits, for BCEWithLogitsLoss.
                resistance_mechanism: (B, num_mechanisms) logits, for CrossEntropyLoss.
                amr_gene_family: (B, num_families) logits, for CrossEntropyLoss.
        """
        shared = self.trunk(x)
        return {
            "drug_class": self.drug_class_head(shared),
            "resistance_mechanism": self.resistance_mechanism_head(shared),
            "amr_gene_family": self.amr_gene_family_head(shared),
        }
