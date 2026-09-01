"""Smoke tests for SoftPromptModule (src/models/soft_prompt.py).

Uses embed_dim=320 (esm2_t6_8M_UR50D hidden size) so tests run in milliseconds
on CPU, matching the 8M-model smoke-test convention used elsewhere in tests/.
"""

import pytest
import torch

from src.models.soft_prompt import NullSoftPrompt, SingleFieldSoftPrompt, SoftPromptModule

EMBED_DIM = 320
NUM_MECHANISMS = 10
NUM_DRUG_CLASSES = 15
N_PROMPT_TOKENS = 2


@pytest.fixture(scope="module")
def soft_prompt() -> SoftPromptModule:
    """SoftPromptModule loaded once per test module."""
    return SoftPromptModule(NUM_MECHANISMS, NUM_DRUG_CLASSES, EMBED_DIM)


def make_inputs(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic dummy mechanism index / multi-hot drug_class inputs.

    Mirrors the shapes AMRDataset actually produces: a scalar long index per
    sample for resistance_mechanism, and a multi-hot float32 vector over the
    full drug_class vocabulary per sample.
    """
    torch.manual_seed(0)
    mechanism = torch.randint(0, NUM_MECHANISMS, (batch_size,))
    drug_classes = (torch.rand(batch_size, NUM_DRUG_CLASSES) > 0.7).float()
    return mechanism, drug_classes


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_shape(self, soft_prompt, batch_size):
        mechanism, drug_classes = make_inputs(batch_size)
        out = soft_prompt(mechanism, drug_classes)
        assert out.shape == (batch_size, N_PROMPT_TOKENS, EMBED_DIM)


# ---------------------------------------------------------------------------
# Output dtype
# ---------------------------------------------------------------------------

class TestOutputDtype:
    def test_dtype_is_float32(self, soft_prompt):
        mechanism, drug_classes = make_inputs(2)
        out = soft_prompt(mechanism, drug_classes)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Output non-zero
# ---------------------------------------------------------------------------

class TestOutputNonZero:
    def test_output_is_non_zero(self, soft_prompt):
        mechanism, drug_classes = make_inputs(2)
        out = soft_prompt(mechanism, drug_classes)
        assert torch.any(out != 0)


# ---------------------------------------------------------------------------
# Trainable parameters
# ---------------------------------------------------------------------------

class TestTrainableParameters:
    def test_all_parameters_trainable(self, soft_prompt):
        params = list(soft_prompt.parameters())
        n_frozen = sum(1 for p in params if not p.requires_grad)
        assert len(params) > 0
        assert n_frozen == 0


# ---------------------------------------------------------------------------
# SingleFieldSoftPrompt (V2 -- Runs 1-3, one categorical conditioning field)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 12


@pytest.fixture(scope="module")
def single_field_soft_prompt() -> SingleFieldSoftPrompt:
    """SingleFieldSoftPrompt loaded once per test module."""
    return SingleFieldSoftPrompt(VOCAB_SIZE, EMBED_DIM)


class TestSingleFieldOutputShape:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_shape(self, single_field_soft_prompt, batch_size):
        torch.manual_seed(0)
        field_index = torch.randint(0, VOCAB_SIZE, (batch_size,))
        out = single_field_soft_prompt(field_index)
        assert out.shape == (batch_size, SingleFieldSoftPrompt.NUM_PROMPT_TOKENS, EMBED_DIM)


class TestSingleFieldOutputDtype:
    def test_dtype_is_float32(self, single_field_soft_prompt):
        field_index = torch.tensor([0, 1])
        out = single_field_soft_prompt(field_index)
        assert out.dtype == torch.float32


class TestSingleFieldTrainableParameters:
    def test_all_parameters_trainable(self, single_field_soft_prompt):
        params = list(single_field_soft_prompt.parameters())
        n_frozen = sum(1 for p in params if not p.requires_grad)
        assert len(params) > 0
        assert n_frozen == 0


# ---------------------------------------------------------------------------
# NullSoftPrompt (V2 negative control -- no conditioning, pairs with Run 3)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def null_soft_prompt() -> NullSoftPrompt:
    """NullSoftPrompt loaded once per test module."""
    return NullSoftPrompt(EMBED_DIM)


class TestNullSoftPromptOutputShape:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_shape(self, null_soft_prompt, batch_size):
        out = null_soft_prompt(batch_size)
        assert out.shape == (batch_size, NullSoftPrompt.NUM_PROMPT_TOKENS, EMBED_DIM)


class TestNullSoftPromptOutputIsZero:
    def test_output_is_all_zero(self, null_soft_prompt):
        out = null_soft_prompt(4)
        assert torch.all(out == 0)


class TestNullSoftPromptNoTrainableParameters:
    def test_no_trainable_parameters(self, null_soft_prompt):
        # The zero token is a registered buffer, not an nn.Parameter -- it
        # must never train, or the negative control stops being a negative
        # control.
        assert len(list(null_soft_prompt.parameters())) == 0
