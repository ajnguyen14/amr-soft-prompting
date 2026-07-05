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

## Partially addressed — flagged for later

3. **Loss weights not config-driven** — `configs/base.yaml` now has a `loss:`
   section (`weight_drug_class`/`weight_resistance_mechanism`/
   `weight_amr_gene_family`, all `1.0`) matching `AMRLoss`'s constructor
   defaults. **Still open:** nothing reads this section yet. A TODO comment in
   `src/training/loss.py` and a note in `docs/STATUS.md`'s Open Questions flag
   that `train.py` must construct `AMRLoss(**config["loss"])` explicitly, or a
   future edit to the config will silently do nothing.

## Deferred — no effect on training correctness today

4. **`input_dim`/N formula duplicated** across `tests/test_classifier.py`,
   `tests/test_forward_pass.py`, and `ESM2Wrapper`'s docstring, with no single
   source of truth. Only becomes a real risk if the soft-prompt token count
   ever changes (V2+ territory). **Plan:** fix when `train.py` is written, by
   adding an `ESM2Wrapper.output_dim` property instead of re-deriving the
   formula a fourth time.
5. **`docs/STATUS.md` test-count arithmetic error** — the breakdown at line 61
   still lists "20 dataset" (should be 18); `22+20+32+6+9+2=91`, not the
   correct `89`. Confirmed via `pytest --collect-only`. Doc-only, zero runtime
   effect. Not yet fixed.
6. **Weaker ESM-2-frozen gradient assertion** in `tests/test_forward_pass.py`
   (`grad is None or grad.abs().sum() == 0`) vs. the stricter `grad is None`
   in `tests/test_esm2_wrapper.py`. The looser clause is currently dead code
   (frozen params never populate `.grad` at all), so it doesn't mask anything
   today.
7. **`AMRLoss`'s `drug_class_labels`→`drug_class` key mapping isn't shared** —
   only matters once `src/eval/evaluate.py` exists and needs the same
   correspondence between predictions and ground truth. Nothing to do until
   that module is written.
8. **Test files missing type hints/docstrings** on `test_*` methods and test
   classes in `tests/test_classifier.py` and `tests/test_forward_pass.py` —
   literal CLAUDE.md violation (no test-file exemption stated for these two
   rules), but pure style, zero functional impact.

## Test-coverage gap (recommend closing before first GPU run, not before writing train.py)

9. **`tests/test_forward_pass.py` never exercises real `AMRDataset`** — it
   hand-builds a batch dict with keys/dtypes chosen to already match what
   `AMRLoss`/`ClassifierHead` expect, rather than running data through
   `AMRDataset` + `DataLoader` collation. This is the same class of bug that
   was caught only by manually re-reading `dataset.py`'s source during the
   `AMRLoss` design (see `docs/sessions/2026-07-05-loss-and-forward-pass.md`).
   Cheap to close on CPU; recommended before spending GPU time on a run that
   could fail on the first real batch over an undetected key/shape mismatch.
