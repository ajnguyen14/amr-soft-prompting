"""Smoke tests for ESM2Wrapper (src/models/esm2_wrapper.py).

All tests use esm2_t6_8M_UR50D (320-dim, 6 layers) so they complete in seconds
on CPU. The two module-scoped fixtures load the model once each and share it
across all tests in that class.
"""

import pytest
import torch

from src.models.esm2_wrapper import ESM2Wrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
EMBED_DIM = 320       # hidden_size for esm2_t6_8M_UR50D
N_PROMPT = 3          # arbitrary small prompt length used throughout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wrapper_internal() -> ESM2Wrapper:
    """ESM2Wrapper in internal mode, loaded once per test module."""
    return ESM2Wrapper(MODEL_NAME, injection_mode="internal")


@pytest.fixture(scope="module")
def wrapper_external() -> ESM2Wrapper:
    """ESM2Wrapper in external mode, loaded once per test module."""
    return ESM2Wrapper(MODEL_NAME, injection_mode="external")


def make_prompt(batch_size: int, n_prompt: int = N_PROMPT) -> torch.Tensor:
    """Deterministic dummy soft prompt of shape (B, N, D)."""
    torch.manual_seed(0)
    return torch.randn(batch_size, n_prompt, EMBED_DIM)


# Sequences of varying length to exercise padding in batched tests.
SEQ_A = "MKAYFIAILT"
SEQ_B = "MKAYFIAILTLFTCIATVVRAQQMSELENRIDSLLNGK"
SEQ_C = "MTKKMNKYNGKKLSRGEPPNFSGQHFMHNKRLLKEIVDKMKAYFIAILTLFTCIATVVR"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_invalid_injection_mode_raises(self):
        with pytest.raises(ValueError, match="injection_mode"):
            ESM2Wrapper(MODEL_NAME, injection_mode="prefix")

    def test_embed_dim_attribute(self, wrapper_internal):
        assert wrapper_internal.embed_dim == EMBED_DIM

    def test_embed_dim_same_for_external(self, wrapper_external):
        assert wrapper_external.embed_dim == EMBED_DIM

    def test_injection_mode_stored(self, wrapper_internal, wrapper_external):
        assert wrapper_internal.injection_mode == "internal"
        assert wrapper_external.injection_mode == "external"


# ---------------------------------------------------------------------------
# output_dim — single source of truth for the mode-dependent output width
# ---------------------------------------------------------------------------

class TestOutputDim:
    def test_internal_mode_ignores_num_prompt_tokens(self, wrapper_internal):
        for n in (1, 2, 5):
            assert wrapper_internal.output_dim(n) == EMBED_DIM

    def test_external_mode_scales_with_num_prompt_tokens(self, wrapper_external):
        for n in (1, 2, 5):
            assert wrapper_external.output_dim(n) == EMBED_DIM + n * EMBED_DIM

    def test_external_mode_matches_actual_forward_output_width(self, wrapper_external):
        with torch.no_grad():
            out = wrapper_external([SEQ_A], make_prompt(1, n_prompt=N_PROMPT))
        assert out.shape[1] == wrapper_external.output_dim(N_PROMPT)


# ---------------------------------------------------------------------------
# Frozen-parameter guarantee
# ---------------------------------------------------------------------------

class TestFrozenParameters:
    def test_zero_trainable_esm_params_internal(self, wrapper_internal):
        n = sum(1 for p in wrapper_internal.esm.parameters() if p.requires_grad)
        assert n == 0, f"Expected 0 trainable ESM-2 params, got {n}"

    def test_zero_trainable_esm_params_external(self, wrapper_external):
        n = sum(1 for p in wrapper_external.esm.parameters() if p.requires_grad)
        assert n == 0, f"Expected 0 trainable ESM-2 params, got {n}"

    def test_zero_trainable_esm_params_total(self, wrapper_internal):
        # Total trainable params in the whole wrapper should be 0 (no soft prompt here).
        n = sum(1 for p in wrapper_internal.parameters() if p.requires_grad)
        assert n == 0

    def test_esm_params_have_no_grad_after_forward(self, wrapper_internal):
        """Confirm ESM-2 gradients don't accumulate even when soft_prompt_vectors do."""
        prompt = make_prompt(1).requires_grad_(True)
        with torch.enable_grad():
            out = wrapper_internal([SEQ_A], prompt)
            out.sum().backward()
        # ESM-2 params must have no grad.
        for name, param in wrapper_internal.esm.named_parameters():
            assert param.grad is None, f"Unexpected gradient on ESM-2 param: {name}"
        # soft_prompt_vectors should have a gradient (they're the learnable part).
        assert prompt.grad is not None


# ---------------------------------------------------------------------------
# Output shapes — internal mode
# ---------------------------------------------------------------------------

class TestInternalModeShapes:
    def test_single_sequence(self, wrapper_internal):
        with torch.no_grad():
            out = wrapper_internal([SEQ_A], make_prompt(1))
        assert out.shape == (1, EMBED_DIM)

    def test_batch_of_two(self, wrapper_internal):
        with torch.no_grad():
            out = wrapper_internal([SEQ_A, SEQ_B], make_prompt(2))
        assert out.shape == (2, EMBED_DIM)

    def test_batch_of_three_variable_lengths(self, wrapper_internal):
        with torch.no_grad():
            out = wrapper_internal([SEQ_A, SEQ_B, SEQ_C], make_prompt(3))
        assert out.shape == (3, EMBED_DIM)

    def test_output_dtype_float32(self, wrapper_internal):
        with torch.no_grad():
            out = wrapper_internal([SEQ_A], make_prompt(1))
        assert out.dtype == torch.float32

    def test_prompt_length_does_not_change_output_dim(self, wrapper_internal):
        """Changing N (number of prompt tokens) must not change the output shape."""
        for n in [1, 5, 10]:
            with torch.no_grad():
                out = wrapper_internal([SEQ_A], make_prompt(1, n_prompt=n))
            assert out.shape == (1, EMBED_DIM), f"Failed for N={n}"

    def test_output_not_all_zeros(self, wrapper_internal):
        """Sanity check — mean-pooled output should be non-trivial."""
        with torch.no_grad():
            out = wrapper_internal([SEQ_A], make_prompt(1))
        assert not torch.all(out == 0.0)


# ---------------------------------------------------------------------------
# Output shapes — external mode
# ---------------------------------------------------------------------------

class TestExternalModeShapes:
    def test_single_sequence(self, wrapper_external):
        with torch.no_grad():
            out = wrapper_external([SEQ_A], make_prompt(1))
        assert out.shape == (1, EMBED_DIM + N_PROMPT * EMBED_DIM)

    def test_batch_of_two(self, wrapper_external):
        with torch.no_grad():
            out = wrapper_external([SEQ_A, SEQ_B], make_prompt(2))
        assert out.shape == (2, EMBED_DIM + N_PROMPT * EMBED_DIM)

    def test_batch_of_three_variable_lengths(self, wrapper_external):
        with torch.no_grad():
            out = wrapper_external([SEQ_A, SEQ_B, SEQ_C], make_prompt(3))
        assert out.shape == (3, EMBED_DIM + N_PROMPT * EMBED_DIM)

    def test_output_dtype_float32(self, wrapper_external):
        with torch.no_grad():
            out = wrapper_external([SEQ_A], make_prompt(1))
        assert out.dtype == torch.float32

    def test_output_dim_scales_with_prompt_tokens(self, wrapper_external):
        """Output width must grow linearly with N."""
        for n in [1, 5, 10]:
            with torch.no_grad():
                out = wrapper_external([SEQ_A], make_prompt(1, n_prompt=n))
            expected = EMBED_DIM + n * EMBED_DIM
            assert out.shape == (1, expected), f"Failed for N={n}"

    def test_esm_slice_matches_standalone_forward(self, wrapper_external):
        """The first embed_dim columns must equal the standalone ESM-2 mean pool."""
        torch.manual_seed(42)
        prompt = make_prompt(1)
        with torch.no_grad():
            out_external = wrapper_external([SEQ_A], prompt)
            # Also run ESM-2 without soft prompt to get the baseline pooled output.
            encoding = wrapper_external._tokenize([SEQ_A])
            esm_out = wrapper_external.esm(**encoding)
            residue_mask = wrapper_external._build_residue_mask(encoding["attention_mask"])
            pooled_direct = wrapper_external._mean_pool(esm_out.last_hidden_state, residue_mask)

        assert torch.allclose(out_external[:, :EMBED_DIM], pooled_direct, atol=1e-5), (
            "First embed_dim columns of external output should equal standalone ESM-2 pool"
        )


# ---------------------------------------------------------------------------
# Residue mask — boundary token exclusion
# ---------------------------------------------------------------------------

class TestResidueMask:
    """Test _build_residue_mask directly for boundary and padding exclusion."""

    def test_cls_excluded(self, wrapper_internal):
        attn = torch.ones(1, 6, dtype=torch.long)
        mask = wrapper_internal._build_residue_mask(attn)
        assert not mask[0, 0].item(), "<cls> at position 0 must be excluded"

    def test_eos_excluded(self, wrapper_internal):
        # All-ones attention_mask: <eos> is at the last position.
        attn = torch.ones(1, 6, dtype=torch.long)
        mask = wrapper_internal._build_residue_mask(attn)
        assert not mask[0, 5].item(), "<eos> at last position must be excluded"

    def test_padding_excluded(self, wrapper_internal):
        # Sequence of 4 real tokens + 2 padding in a batch of length 6.
        attn = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
        mask = wrapper_internal._build_residue_mask(attn)
        assert not mask[0, 4].item(), "Padding position 4 must be excluded"
        assert not mask[0, 5].item(), "Padding position 5 must be excluded"

    def test_residue_positions_included(self, wrapper_internal):
        # 6-token sequence: cls, r1, r2, r3, r4, eos
        attn = torch.ones(1, 6, dtype=torch.long)
        mask = wrapper_internal._build_residue_mask(attn)
        for pos in [1, 2, 3, 4]:
            assert mask[0, pos].item(), f"Residue at position {pos} must be included"

    def test_batch_with_variable_lengths(self, wrapper_internal):
        # Batch of 2: lengths 6 and 3 (padded to 6).
        attn = torch.tensor([
            [1, 1, 1, 1, 1, 1],   # cls, r, r, r, r, eos
            [1, 1, 1, 0, 0, 0],   # cls, r, eos, pad, pad, pad
        ], dtype=torch.long)
        mask = wrapper_internal._build_residue_mask(attn)

        # First sequence: positions 1..4 are residues.
        assert not mask[0, 0]  # cls
        assert not mask[0, 5]  # eos
        for pos in [1, 2, 3, 4]:
            assert mask[0, pos], f"Seq 0, position {pos} should be residue"

        # Second sequence: only position 1 is a residue.
        assert not mask[1, 0]  # cls
        assert not mask[1, 2]  # eos
        assert mask[1, 1]      # only residue
        assert not mask[1, 3] and not mask[1, 4] and not mask[1, 5]  # padding

    def test_minimal_sequence(self, wrapper_internal):
        # One real token besides cls and eos: cls, r, eos.
        attn = torch.tensor([[1, 1, 1]], dtype=torch.long)
        mask = wrapper_internal._build_residue_mask(attn)
        assert not mask[0, 0]  # cls
        assert not mask[0, 2]  # eos
        assert mask[0, 1]      # single residue

    def test_returns_bool_tensor(self, wrapper_internal):
        attn = torch.ones(2, 5, dtype=torch.long)
        mask = wrapper_internal._build_residue_mask(attn)
        assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# Mean pooling helper
# ---------------------------------------------------------------------------

class TestMeanPool:
    def test_uniform_mask(self, wrapper_internal):
        # All positions included — result should be the elementwise mean.
        hidden = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])  # (1, 3, 2)
        mask = torch.tensor([[1, 1, 1]])
        result = wrapper_internal._mean_pool(hidden, mask)
        expected = torch.tensor([[3., 4.]])  # mean of [1,3,5], [2,4,6]
        assert torch.allclose(result, expected)

    def test_single_included_position(self, wrapper_internal):
        hidden = torch.tensor([[[1., 1.], [7., 8.], [3., 3.]]])
        mask = torch.tensor([[0, 1, 0]])
        result = wrapper_internal._mean_pool(hidden, mask)
        assert torch.allclose(result, torch.tensor([[7., 8.]]))

    def test_no_nan_with_all_zeros_mask(self, wrapper_internal):
        # All-zero mask: guard against division by zero → should return zeros, not NaN.
        hidden = torch.ones(1, 4, 8)
        mask = torch.zeros(1, 4)
        result = wrapper_internal._mean_pool(hidden, mask)
        assert not torch.isnan(result).any()
        assert torch.all(result == 0.0)

    def test_batch_pooling_is_independent(self, wrapper_internal):
        # Each batch item should pool independently based on its own mask.
        hidden = torch.tensor([
            [[10., 10.], [20., 20.], [30., 30.]],  # batch 0
            [[1., 1.],   [2., 2.],   [3., 3.]],    # batch 1
        ])
        mask = torch.tensor([
            [1, 0, 0],  # batch 0: only position 0
            [0, 0, 1],  # batch 1: only position 2
        ])
        result = wrapper_internal._mean_pool(hidden, mask)
        assert torch.allclose(result[0], torch.tensor([10., 10.]))
        assert torch.allclose(result[1], torch.tensor([3., 3.]))

    def test_output_shape(self, wrapper_internal):
        hidden = torch.randn(4, 12, EMBED_DIM)
        mask = torch.ones(4, 12, dtype=torch.long)
        result = wrapper_internal._mean_pool(hidden, mask)
        assert result.shape == (4, EMBED_DIM)
