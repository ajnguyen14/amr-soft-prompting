"""MLP classification head: shared trunk feeding a single amr_gene_family output head."""

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


class SingleTargetClassifierHead(nn.Module):
    """Shared-trunk MLP producing logits for one configurable V2 target.

    Generalizes ClassifierHead (V1, hardcoded to amr_gene_family) to any
    single Run 1-3 prediction target (drug_class, resistance_mechanism, or
    amr_gene_family -- see CLAUDE.md's Single-Head Architecture table). Kept
    as a separate class rather than modifying ClassifierHead in place, since
    V1's two trained checkpoints' state_dicts are keyed to ClassifierHead's
    existing `amr_gene_family_head` attribute name.

    Args:
        input_dim: Width of the incoming representation from ESM2Wrapper.
            Mode dependent -- embed_dim for 'internal' injection, or
            embed_dim + N * embed_dim for 'external' injection.
        hidden_dim: Width of the shared trunk's hidden layer. From config.
        dropout: Dropout probability applied after the trunk's ReLU. From config.
        target_name: Name of the prediction target (e.g. 'drug_class'), used
            as the forward() output dict's only key.
        num_classes: Size of the target's label vocabulary.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        target_name: str,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.target_name = target_name

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the shared trunk then the configured target's head.

        Args:
            x: Tensor of shape (B, input_dim) -- pooled representation from
                ESM2Wrapper (internal or external injection mode).

        Returns:
            Dict with one key, self.target_name: (B, num_classes) logits.
        """
        shared = self.trunk(x)
        return {self.target_name: self.head(shared)}
