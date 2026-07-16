"""Frozen ESM-2 backbone with internal and external soft prompt injection modes."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel


_VALID_INJECTION_MODES = frozenset({"internal", "external"})


class ESM2Wrapper(nn.Module):
    """Frozen ESM-2 backbone supporting two soft prompt injection strategies.

    Both modes share the same frozen ESM-2 backbone and differ only in where
    soft_prompt_vectors are introduced relative to the transformer layers.

    Args:
        model_name: HuggingFace model identifier, e.g. 'facebook/esm2_t6_8M_UR50D'.
        injection_mode: How soft prompts are combined with ESM-2 representations.
            'internal' — prepend soft prompt tokens before the transformer encoder;
                gradients from the loss flow back through all ESM-2 layers to
                soft_prompt_vectors (but NOT to ESM-2 parameters, which are frozen).
            'external' — concatenate flattened soft prompt with the mean-pooled
                ESM-2 output; ESM-2 and the soft prompt are fully decoupled.

    Attributes:
        embed_dim: ESM-2 hidden size. soft_prompt_vectors must have this as their
            last dimension. Also the output dimension for internal mode.
        injection_mode: The active injection strategy.

    Output shapes (from forward()):
        internal mode → (B, embed_dim)
        external mode → (B, embed_dim + num_prompt_tokens * embed_dim)
        The classifier head must be built to match whichever mode is active.
    """

    def __init__(self, model_name: str, injection_mode: str = "internal") -> None:
        super().__init__()

        if injection_mode not in _VALID_INJECTION_MODES:
            raise ValueError(
                f"injection_mode must be one of {_VALID_INJECTION_MODES!r}, got {injection_mode!r}"
            )

        self.injection_mode = injection_mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm = EsmModel.from_pretrained(model_name)

        # Freeze ALL ESM-2 parameters — weights are never updated regardless of injection_mode.
        for param in self.esm.parameters():
            param.requires_grad = False

        # Hard assertion: zero trainable params in the backbone, always.
        n_trainable = sum(1 for p in self.esm.parameters() if p.requires_grad)
        assert n_trainable == 0, (
            f"ESM-2 backbone has {n_trainable} trainable parameter(s) — expected 0. "
            "Check that all ESM-2 parameters have requires_grad=False."
        )

        self.embed_dim: int = self.esm.config.hidden_size

        if injection_mode == "internal":
            # Internal mode backprops through all encoder layers to reach
            # soft_prompt_vectors, retaining every layer's activations for a
            # full 650M-parameter, 33-layer stack — OOMs at batch_size 32.
            # Enables HF's per-layer checkpointing machinery on each EsmLayer;
            # actually engaging it at call time additionally requires flipping
            # each layer's `.training` flag (see _forward_internal), since
            # this codebase keeps ESM-2 in eval() throughout.
            self.esm.gradient_checkpointing_enable()

    @property
    def device(self) -> torch.device:
        """Device the ESM-2 model is on."""
        return next(self.esm.parameters()).device

    def output_dim(self, num_prompt_tokens: int) -> int:
        """Width of forward()'s output for this wrapper's injection_mode.

        Single source of truth for the mode-dependent output width formula
        (see forward()'s docstring), so callers building a ClassifierHead
        don't need to re-derive `embed_dim + N * embed_dim` themselves.

        Args:
            num_prompt_tokens: N, the number of soft prompt tokens that will
                be passed to forward() (e.g. SoftPromptModule.NUM_PROMPT_TOKENS).

        Returns:
            embed_dim for 'internal' mode; embed_dim + num_prompt_tokens *
            embed_dim for 'external' mode.
        """
        if self.injection_mode == "internal":
            return self.embed_dim
        return self.embed_dim + num_prompt_tokens * self.embed_dim

    def forward(
        self,
        sequences: list[str],
        soft_prompt_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Run sequences through frozen ESM-2 conditioned by soft prompt vectors.

        Tokenization happens here — dataset.py returns raw amino acid strings and
        defers all tokenization to this method.

        Args:
            sequences: List of B raw amino acid strings. Variable-length sequences
                are padded to the longest in the batch by the ESM-2 tokenizer.
            soft_prompt_vectors: Tensor of shape (B, N, D) where N is the number of
                prompt tokens and D must equal self.embed_dim. Produced by
                soft_prompt.py — this wrapper only receives and uses them.
                Must be on the same device as this module.

        Returns:
            internal mode: Tensor of shape (B, embed_dim). Mean-pooled over residue
                positions only (prompt tokens, <cls>, <eos>, and padding excluded).
            external mode: Tensor of shape (B, embed_dim + N * embed_dim). The first
                embed_dim features are the ESM-2 mean-pool; the remaining N * embed_dim
                are the flattened soft prompt. The classifier head must account for
                this combined width.

        # TODO: soft_prompt_vectors are assumed to have D == self.embed_dim. If
        #       soft_prompt.py produces a different embedding dimension, a projection
        #       will be needed either here or in the classifier. Confirm interface with
        #       soft_prompt.py design before finalising.
        """
        if self.injection_mode == "internal":
            return self._forward_internal(sequences, soft_prompt_vectors)
        return self._forward_external(sequences, soft_prompt_vectors)

    # ------------------------------------------------------------------
    # Private: injection-mode implementations
    # ------------------------------------------------------------------

    def _forward_internal(
        self,
        sequences: list[str],
        soft_prompt_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Prepend soft prompt tokens before ESM-2's transformer layers.

        soft_prompt_vectors are inserted at the front of the token sequence so
        that every transformer layer attends over prompt + residue tokens jointly.
        Gradients flow from the output back through all encoder layers to
        soft_prompt_vectors (ESM-2 parameters remain frozen throughout).

        Mean pooling covers residue positions only: prompt positions, <cls>, <eos>,
        and padding are all excluded.
        """
        B, N, _D = soft_prompt_vectors.shape

        encoding = self._tokenize(sequences)
        input_ids = encoding["input_ids"]            # (B, L)
        attention_mask = encoding["attention_mask"]  # (B, L)

        # Pre-process word embeddings through ESM-2's embedding layer so that
        # token_dropout scaling (applied during training and inference when
        # input_ids are provided) is consistent with the external-mode path.
        # When inputs_embeds is passed to EsmModel.forward(), the embedding
        # layer is bypassed entirely — so we call it here explicitly.
        word_embeds = self.esm.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # (B, L, D)

        # Prepend soft prompt tokens along the sequence dimension.
        combined_embeds = torch.cat([soft_prompt_vectors, word_embeds], dim=1)  # (B, N+L, D)

        # Extend attention mask: prompt positions are always fully attended to.
        prompt_ones = torch.ones(B, N, dtype=attention_mask.dtype, device=self.device)
        extended_attn = torch.cat([prompt_ones, attention_mask], dim=1)  # (B, N+L)

        # Run through the frozen ESM-2 encoder. inputs_embeds bypasses the
        # embedding layer (already applied above); RoPE position IDs are
        # auto-generated as 0..N+L-1 for the combined sequence.
        #
        # HF's per-layer gradient checkpointing (enabled in __init__) only
        # activates on a layer when that layer's own `.training` is True —
        # but this codebase always calls esm.eval() (see train.py) to keep
        # frozen ESM-2's dropout off, which recursively sets `.training =
        # False` on every submodule including each EsmLayer. So we flip just
        # the EsmLayer submodules' `.training` back to True here — a direct
        # attribute set, NOT a recursive `.train()` call — so only the
        # per-layer checkpointing gate engages; every dropout submodule stays
        # at whatever eval() left it. This is numerically inert regardless:
        # hidden_dropout_prob and attention_probs_dropout_prob are both 0.0
        # for all four ESM-2 variants this project uses (8M/150M/650M/3B),
        # confirmed via AutoConfig for each.
        #
        # Without this, internal mode backprops through all encoder layers to
        # reach soft_prompt_vectors, retaining every layer's activations for
        # a full 650M-parameter, 33-layer stack — OOMs at batch_size 32.
        for layer in self.esm.encoder.layer:
            layer.training = True
        try:
            outputs = self.esm(inputs_embeds=combined_embeds, attention_mask=extended_attn)
        finally:
            for layer in self.esm.encoder.layer:
                layer.training = False
        hidden_states = outputs.last_hidden_state  # (B, N+L, D)

        # Build pooling mask in the extended sequence space:
        # prompt positions → 0, residue positions → 1, <cls>/<eos>/pad → 0.
        residue_mask = self._build_residue_mask(attention_mask)  # (B, L)
        prompt_zeros = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        extended_residue_mask = torch.cat([prompt_zeros, residue_mask], dim=1)  # (B, N+L)

        return self._mean_pool(hidden_states, extended_residue_mask)  # (B, D)

    def _forward_external(
        self,
        sequences: list[str],
        soft_prompt_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate flattened soft prompt with mean-pooled ESM-2 output.

        ESM-2 processes sequences independently of the soft prompt. The two signals
        are joined only at the output, giving the classifier head access to both
        without any interaction in the encoder.
        """
        B, N, D = soft_prompt_vectors.shape

        encoding = self._tokenize(sequences)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Run through frozen ESM-2 normally (embedding layer + encoder).
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, D)

        # Mean pool over residue tokens only.
        residue_mask = self._build_residue_mask(attention_mask)  # (B, L)
        pooled = self._mean_pool(hidden_states, residue_mask)     # (B, D)

        # Flatten soft prompt and concatenate with pooled representation.
        # The classifier receives both signals independently — no projection here.
        prompt_flat = soft_prompt_vectors.reshape(B, N * D)           # (B, N*D)
        return torch.cat([pooled, prompt_flat], dim=1)                # (B, D + N*D)

    # ------------------------------------------------------------------
    # Private: tokenization, masking, pooling
    # ------------------------------------------------------------------

    def _tokenize(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a list of amino acid sequences and move tensors to device.

        Args:
            sequences: List of raw amino acid strings.

        Returns:
            Dict with 'input_ids' and 'attention_mask', both on self.device.
        """
        encoding = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    def _build_residue_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Build a boolean mask selecting only amino acid residue positions.

        Excludes three categories from pooling:
        - <cls> token at position 0 (always present in ESM-2 sequences).
        - <eos> token at the last real (non-padding) position.
        - Padding tokens (0s in the original attention_mask).

        Args:
            attention_mask: (B, L) int tensor — 1 for real tokens, 0 for padding.

        Returns:
            (B, L) bool tensor — True only at amino acid residue positions.
        """
        residue_mask = attention_mask.bool().clone()

        # Exclude <cls> at position 0.
        residue_mask[:, 0] = False

        # Exclude <eos>: the last non-padding token in each sequence.
        # seq_lengths[b] = total real tokens (including <cls> and <eos>).
        seq_lengths = attention_mask.sum(dim=1).long()                        # (B,)
        L = attention_mask.size(1)
        arange = torch.arange(L, device=attention_mask.device).unsqueeze(0)  # (1, L)
        eos_positions = (seq_lengths - 1).clamp(min=0).unsqueeze(1)          # (B, 1)
        eos_mask = arange == eos_positions                                    # (B, L)
        residue_mask = residue_mask & ~eos_mask

        return residue_mask  # (B, L), True at residue positions only

    def _mean_pool(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a masked mean over the sequence dimension.

        Args:
            hidden_states: (B, L, D) float tensor from the ESM-2 encoder.
            mask: (B, L) bool or long tensor — True/1 for positions to include.

        Returns:
            (B, D) mean-pooled tensor. Sequences with zero valid positions (should
            not occur in practice) return a zero vector rather than NaN.
        """
        mask_f = mask.float().unsqueeze(-1)                    # (B, L, 1)
        summed = (hidden_states * mask_f).sum(dim=1)           # (B, D)
        counts = mask_f.sum(dim=1).clamp(min=1.0)             # (B, 1) — guard div/0
        return summed / counts                                  # (B, D)
