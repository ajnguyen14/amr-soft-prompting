"""MLP classification head: shared trunk with three task-specific output heads."""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Shared-trunk MLP producing amr_gene_family logits.

    A single hidden layer (trunk) feeds one linear head predicting
    amr_gene_family (single-label, paired with CrossEntropyLoss downstream).

    resistance_mechanism and drug_class are deliberately not predicted here:
    both are fed into SoftPromptModule as conditioning input, so predicting
    them back out would just be decoding their own soft-prompt embedding
    rather than learning anything from the ESM-2 sequence representation (see
    docs/STATUS.md's label-leakage note). amr_gene_family is never fed into
    the soft prompt, so it's the only task free of that leakage.

    Args:
        input_dim: Width of the incoming representation from ESM2Wrapper. Mode
            dependent — embed_dim for 'internal' injection, or
            embed_dim + N * embed_dim for 'external' injection. Passed in by the
            caller rather than inferred here.
        hidden_dim: Width of the shared trunk's hidden layer. From config.
        dropout: Dropout probability applied after the trunk's ReLU. From config.
        num_families: Size of the amr_gene_family vocabulary, from
            card_parser.get_label_vocabularies().
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_families: int,
    ) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.amr_gene_family_head = nn.Linear(hidden_dim, num_families)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the shared trunk then the amr_gene_family head.

        Args:
            x: Tensor of shape (B, input_dim) — pooled representation from
                ESM2Wrapper (internal or external injection mode).

        Returns:
            Dict with key:
                amr_gene_family: (B, num_families) logits, for CrossEntropyLoss.
        """
        shared = self.trunk(x)
        return {
            "amr_gene_family": self.amr_gene_family_head(shared),
        }
