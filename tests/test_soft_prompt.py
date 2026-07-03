"""Smoke tests for SoftPromptModule (src/models/soft_prompt.py).

Uses embed_dim=320 (esm2_t6_8M_UR50D hidden size) so tests run in milliseconds
on CPU, matching the 8M-model smoke-test convention used elsewhere in tests/.
"""

import pytest
import torch

from src.models.soft_prompt import SoftPromptModule

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
