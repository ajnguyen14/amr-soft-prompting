# Code Review Log — 2026-07-05 (classifier.py + loss.py + test_forward_pass.py)

`/code-review` (high effort, 8 finder angles + recall-biased verification) run
against `git diff HEAD~2...HEAD` — the session that added `ClassifierHead`,
`AMRLoss`, and the full-pipeline integration test. 9 findings survived
verification. This doc tracks what's been fixed and what's still open so we
don't lose track of the deferred items.

## Fixed

1. **BCEWithLogitsLoss/CrossEntropyLoss reduction mismatch** (`src/training/loss.py`)
   — `BCEWithLogitsLoss`'s default `mean` divides by `batch_size * num_classes`,
   while `CrossEntropyLoss`'s default `mean` divides by `batch_size` only, so the
   documented "equal weighting (α=β=γ=1)" didn't actually give the three tasks
   comparable gradient magnitude. **Fix:** `BCEWithLogitsLoss(reduction="sum")`
   + manual `/batch_size`, putting all three terms on the same per-sample scale.
   Comment added explaining why.
2. **RNG seed order in `tests/test_forward_pass.py`** — `torch.manual_seed(0)`
   was only called inside `make_batch()`, after `SoftPromptModule`/
   `ClassifierHead` were already constructed, so model weight init wasn't
   seeded. **Fix:** seed moved to the top of `test_forward_and_backward`,
   before any module is constructed.
3. **Loss weights not config-driven** — `configs/base.yaml` has a `loss:`
   section (`weight_drug_class`/`weight_resistance_mechanism`/
   `weight_amr_gene_family`, all `1.0`). **Fixed:** `src/training/train.py`'s
   `build_models()` now constructs `AMRLoss(**config["loss"])` explicitly, so
   the config is the actual source of truth, not a coincidentally-matching
   Python default.
4. **`input_dim`/N formula duplicated** across `tests/test_classifier.py`,
   `tests/test_forward_pass.py`, and `ESM2Wrapper`'s docstring, with no single
   source of truth. **Fixed:** added `ESM2Wrapper.output_dim(num_prompt_tokens)`
   and `SoftPromptModule.NUM_PROMPT_TOKENS` (=2) as the single source of truth
   for both halves of the formula; `train.py` and `test_forward_pass.py` now
   call `esm2.output_dim(SoftPromptModule.NUM_PROMPT_TOKENS)` instead of
   re-deriving it. (`test_classifier.py`'s hardcoded constant was left alone —
   that test deliberately avoids constructing a real `ESM2Wrapper` to stay fast
   and dependency-free, so it isn't testing the production formula anyway.)
5. **`tests/test_forward_pass.py` never exercises real `AMRDataset`** —
   substantially addressed, not by changing `test_forward_pass.py` itself, but
   because `tests/test_train.py` now runs the real `load_card_dataset` →
   `AMRDataset` → `DataLoader` → training loop path end-to-end against a
   synthetic CARD file on disk, closing the actual gap (a real `AMRDataset`
   integration was untested anywhere) via a more appropriate test file than the
   one that originally surfaced it.

## Deferred — no effect on training correctness today

6. **`docs/STATUS.md` test-count arithmetic error** — a past revision's
   breakdown listed "20 dataset" (should be 18), so the stated per-category
   sum didn't match the stated total. Doc-only, zero runtime effect; STATUS.md
   has since been rewritten multiple times and should be double-checked for
   arithmetic the next time it's touched, but isn't worth a dedicated fix pass.
7. **Weaker ESM-2-frozen gradient assertion** in `tests/test_forward_pass.py`
   (`grad is None or grad.abs().sum() == 0`) vs. the stricter `grad is None`
   in `tests/test_esm2_wrapper.py`. The looser clause is currently dead code
   (frozen params never populate `.grad` at all), so it doesn't mask anything
   today.
8. **`AMRLoss`'s `drug_class_labels`→`drug_class` key mapping isn't shared** —
   only matters once `src/eval/evaluate.py` (the fuller per-class F1 /
   confusion-matrix holdout eval, not the `src/eval/metrics.py` per-epoch
   helper added this session) exists and needs the same correspondence between
   predictions and ground truth. Nothing to do until that module is written.
9. **Test files missing type hints/docstrings** on `test_*` methods and test
   classes in `tests/test_classifier.py` and `tests/test_forward_pass.py` —
   literal CLAUDE.md violation (no test-file exemption stated for these two
   rules), but pure style, zero functional impact.
