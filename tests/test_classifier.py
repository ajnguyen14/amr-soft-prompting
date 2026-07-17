"""Smoke tests for ClassifierHead (src/models/classifier.py).

Uses small dummy dims so tests run in milliseconds on CPU, matching the
smoke-test convention used elsewhere in tests/ (see test_soft_prompt.py).
"""

import pytest
import torch

from src.models.classifier import ClassifierHead

INPUT_DIM_INTERNAL = 320  # embed_dim, as in 'internal' injection mode
N_PROMPT_TOKENS = 2
INPUT_DIM_EXTERNAL = INPUT_DIM_INTERNAL + N_PROMPT_TOKENS * INPUT_DIM_INTERNAL  # 'external' mode
HIDDEN_DIM = 64
NUM_FAMILIES = 20
DROPOUT = 0.1


def make_classifier(input_dim: int) -> ClassifierHead:
    """Build a ClassifierHead with the given input width and fixed dummy dims."""
    return ClassifierHead(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        num_families=NUM_FAMILIES,
    )


# ---------------------------------------------------------------------------
# Output shape — internal mode input width
# ---------------------------------------------------------------------------

class TestOutputShapeInternal:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_shape(self, batch_size):
        classifier = make_classifier(INPUT_DIM_INTERNAL)
        x = torch.randn(batch_size, INPUT_DIM_INTERNAL)
        out = classifier(x)
        assert out["amr_gene_family"].shape == (batch_size, NUM_FAMILIES)


# ---------------------------------------------------------------------------
# Output shape — external mode input width
# ---------------------------------------------------------------------------

class TestOutputShapeExternal:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_shape(self, batch_size):
        classifier = make_classifier(INPUT_DIM_EXTERNAL)
        x = torch.randn(batch_size, INPUT_DIM_EXTERNAL)
        out = classifier(x)
        assert out["amr_gene_family"].shape == (batch_size, NUM_FAMILIES)


# ---------------------------------------------------------------------------
# Output dtype
# ---------------------------------------------------------------------------

class TestOutputDtype:
    def test_dtype_is_float32(self):
        classifier = make_classifier(INPUT_DIM_INTERNAL)
        x = torch.randn(2, INPUT_DIM_INTERNAL)
        out = classifier(x)
        assert out["amr_gene_family"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Output keys
# ---------------------------------------------------------------------------

class TestOutputKeys:
    def test_returns_only_amr_gene_family_head(self):
        classifier = make_classifier(INPUT_DIM_INTERNAL)
        x = torch.randn(2, INPUT_DIM_INTERNAL)
        out = classifier(x)
        assert set(out.keys()) == {"amr_gene_family"}


# ---------------------------------------------------------------------------
# Trainable parameters
# ---------------------------------------------------------------------------

class TestTrainableParameters:
    def test_all_parameters_trainable(self):
        classifier = make_classifier(INPUT_DIM_INTERNAL)
        params = list(classifier.parameters())
        n_frozen = sum(1 for p in params if not p.requires_grad)
        assert len(params) > 0
        assert n_frozen == 0
