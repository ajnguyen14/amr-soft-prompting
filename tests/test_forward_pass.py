"""End-to-end integration smoke test for the full V1 model stack.

Wires ESM2Wrapper -> SoftPromptModule -> ClassifierHead -> AMRLoss together and
confirms the whole pipeline runs forward and backward without error. Uses the
8M model (facebook/esm2_t6_8M_UR50D) with batch size 2 on CPU, per CLAUDE.md's
smoke-test convention and minimum forward-pass checklist.
"""

import pytest
import torch

from src.models.esm2_wrapper import ESM2Wrapper
from src.models.soft_prompt import SoftPromptModule
from src.models.classifier import ClassifierHead
from src.training.loss import AMRLoss

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
EMBED_DIM = 320       # hidden_size for esm2_t6_8M_UR50D
HIDDEN_DIM = 64
NUM_MECHANISMS = 5  # SoftPromptModule conditioning input only, no longer a classifier head
NUM_DRUG_CLASSES = 10  # SoftPromptModule conditioning input only, no longer a classifier head
NUM_FAMILIES = 20
DROPOUT = 0.1
BATCH_SIZE = 2

SEQUENCES = ["MKAYFIAILT", "MKAYFIAILTLFTCIATVVRAQQMSELENRIDSLLNGK"]


def make_batch() -> dict[str, torch.Tensor]:
    """Dummy batch mirroring AMRDataset's collated output shapes.

    Does not seed the RNG itself -- callers seed once, before any module
    construction, so both model weight init and this batch are reproducible
    from the same fixed point (see test_forward_and_backward).
    """
    return {
        "drug_class_labels": (torch.rand(BATCH_SIZE, NUM_DRUG_CLASSES) > 0.7).float(),
        "resistance_mechanism": torch.randint(0, NUM_MECHANISMS, (BATCH_SIZE,)),
        "amr_gene_family": torch.randint(0, NUM_FAMILIES, (BATCH_SIZE,)),
    }


@pytest.fixture(scope="module")
def esm2(request) -> ESM2Wrapper:
    """ESM2Wrapper parametrized by injection mode, cached once per mode per module."""
    return ESM2Wrapper(MODEL_NAME, injection_mode=request.param)


class TestFullPipeline:
    """Forward and backward pass through the full V1 model stack."""

    @pytest.mark.parametrize("esm2", ["internal", "external"], indirect=True)
    def test_forward_and_backward(self, esm2):
        # Seed before constructing any module so weight init is reproducible too,
        # not just the dummy batch below.
        torch.manual_seed(0)
        soft_prompt = SoftPromptModule(NUM_MECHANISMS, NUM_DRUG_CLASSES, EMBED_DIM)
        classifier = ClassifierHead(
            input_dim=esm2.output_dim(SoftPromptModule.NUM_PROMPT_TOKENS),
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT,
            num_families=NUM_FAMILIES,
        )
        loss_fn = AMRLoss()
        batch = make_batch()

        soft_prompt_vectors = soft_prompt(
            batch["resistance_mechanism"], batch["drug_class_labels"]
        )
        pooled = esm2(SEQUENCES, soft_prompt_vectors)
        logits = classifier(pooled)
        losses = loss_fn(logits, batch)

        # --- forward assertions ---
        assert set(losses.keys()) == {"amr_gene_family", "total"}
        assert losses["total"].dim() == 0

        # --- backward assertions ---
        losses["total"].backward()

        for name, param in soft_prompt.named_parameters():
            assert param.grad is not None, f"soft_prompt.{name} received no gradient"

        for name, param in classifier.named_parameters():
            assert param.grad is not None, f"classifier.{name} received no gradient"

        for name, param in esm2.esm.named_parameters():
            assert param.grad is None or param.grad.abs().sum() == 0, (
                f"ESM-2 param {name} unexpectedly received a nonzero gradient"
            )


class TestGradientCheckpointing:
    """Regression guard for the internal-mode OOM fix (see esm2_wrapper.py).

    Internal mode backprops through all encoder layers to reach
    soft_prompt_vectors, which OOMs on the real 650M model at the configured
    batch size without per-layer gradient checkpointing. The fix flips each
    EsmLayer's `.training` flag (not a recursive `.train()` call) to engage
    HF's built-in checkpointing while leaving dropout untouched -- these
    tests guard the two ways that could silently break.
    """

    def test_enabled_only_for_internal_mode(self):
        internal = ESM2Wrapper(MODEL_NAME, injection_mode="internal")
        external = ESM2Wrapper(MODEL_NAME, injection_mode="external")
        assert internal.esm.encoder.gradient_checkpointing is True
        assert external.esm.encoder.gradient_checkpointing is False

    def test_layer_training_flag_resets_after_forward(self):
        torch.manual_seed(0)
        esm2 = ESM2Wrapper(MODEL_NAME, injection_mode="internal")
        soft_prompt = SoftPromptModule(NUM_MECHANISMS, NUM_DRUG_CLASSES, EMBED_DIM)
        batch = make_batch()
        soft_prompt_vectors = soft_prompt(
            batch["resistance_mechanism"], batch["drug_class_labels"]
        )

        esm2(SEQUENCES, soft_prompt_vectors)

        assert all(not layer.training for layer in esm2.esm.encoder.layer), (
            "EsmLayer.training should reset to False after forward() returns -- "
            "it's only flipped True transiently to engage per-layer checkpointing"
        )
