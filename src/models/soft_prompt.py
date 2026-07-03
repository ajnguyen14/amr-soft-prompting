"""Soft prompt module encoding CARD metadata as conditioning vectors for ESM-2."""

import torch
import torch.nn as nn


class SoftPromptModule(nn.Module):
    """Encodes CARD metadata as soft prompt tokens for conditioning ESM-2.

    Takes a resistance mechanism (single integer index per sample) and drug
    classes (multi-hot over the drug class vocabulary, matching the encoding
    produced by AMRDataset) and returns 2 soft prompt tokens of shape
    (B, 2, embed_dim), ready to be prepended to ESM-2's input sequence by
    ESM2Wrapper.

    Args:
        num_mechanisms: Size of the resistance mechanism vocabulary.
        num_drug_classes: Size of the drug class vocabulary.
        embed_dim: Output embedding dimension; must equal ESM2Wrapper.embed_dim
            (1280 for the 650M training model) so no projection layer is needed.
    """

    def __init__(self, num_mechanisms: int, num_drug_classes: int, embed_dim: int) -> None:
        super().__init__()
        # Learned lookup table: maps a resistance mechanism index to a dense
        # embed_dim vector. One row per mechanism in the vocabulary.
        self.mechanism_embedding = nn.Embedding(num_mechanisms, embed_dim)
        # Learned lookup table: maps a drug class index to a dense embed_dim
        # vector. One row per drug class in the vocabulary. Looked up via
        # matrix multiplication against the multi-hot input below rather than
        # index-based nn.Embedding lookup, since drug class is multi-label.
        self.drug_class_embedding = nn.Embedding(num_drug_classes, embed_dim)

    def forward(self, mechanism: torch.Tensor, drug_classes: torch.Tensor) -> torch.Tensor:
        """Encode CARD metadata into 2 soft prompt tokens.

        Args:
            mechanism: Long tensor of shape (B,) — the resistance mechanism
                index for each sample.
            drug_classes: Float tensor of shape (B, num_drug_classes) — the
                multi-hot drug class vector for each sample, as produced by
                AMRDataset (1.0 for active classes, 0.0 otherwise).

        Returns:
            Tensor of shape (B, 2, embed_dim): token 0 is the mechanism
            embedding, token 1 is the sum-pooled drug class embedding.
        """
        mechanism_vector = self.mechanism_embedding(mechanism)
        # drug_classes is multi-hot, not a set of indices, so we can't index
        # into the embedding table directly. Instead multiply the multi-hot
        # vector against the embedding weight matrix: (B, num_drug_classes) @
        # (num_drug_classes, embed_dim) -> (B, embed_dim). This is exactly
        # sum pooling over the embeddings of the active classes — sum (not
        # mean) preserves the additive signal across co-occurring drug
        # classes, so a gene with many active classes produces a
        # larger-magnitude vector than one with a single class; mean pooling
        # would dilute that signal away.
        drug_class_vector = drug_classes @ self.drug_class_embedding.weight
        # Stack (not cat) along a new dimension: ESM2Wrapper expects soft
        # prompt tokens as (B, N, D) — separate sequence positions it can
        # prepend to the residue tokens. cat would instead flatten the two
        # vectors into a (B, 2*embed_dim) tensor with no token structure.
        soft_prompts = torch.stack([mechanism_vector, drug_class_vector], dim=1)
        return soft_prompts
