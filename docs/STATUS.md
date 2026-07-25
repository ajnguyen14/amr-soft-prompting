# AMR Soft Prompting — Project Status
_Last updated: 2026-07-20_

## Current Version

**V1 — Core Pipeline**, code complete, retrained, and evaluated. Both
ablation checkpoints are now trained against the fixed (non-leaky)
classifier and have holdout numbers to show for it.

## Completion

**V1 is functionally complete end-to-end**: both ablation configs
(`configs/gpu_server_internal.yaml`, `configs/gpu_server_external.yaml`) have
been retrained for the full 50 epochs against the post-label-leakage-fix
classifier, and both checkpoints have been scored on the CARD test holdout
via `src.eval.evaluate`. What remains is analysis/write-up (poster) rather
than pipeline work.

## What's Working

Everything from the prior entry, plus:

- **GPU server (`sjsu`) CUDA is confirmed working.** The 570/580 driver
  mismatch that previously blocked training/650M inference on this host is
  resolved — `nvidia-smi` reports driver 580.159.03 and
  `torch.cuda.is_available()` returns True. Both retraining runs below and
  today's evaluation ran on this host without issue.
- **Both ablations retrained against the fixed `ClassifierHead`.** Full 50
  epochs each, `batch_size: 24` in both configs (see note under Open
  Questions), logged to wandb as `lunar-eon-7` (internal, run `i7o4eg5n`) and
  `fine-flower-8` (external, run `2rr2h1f9`). Best-val checkpoints saved at
  epoch 47 (internal) and epoch 36 (external).
- **Holdout evaluation run for both**, via
  `python -m src.eval.evaluate --config configs/gpu_server_{internal,external}.yaml`:

  | Injection mode | Checkpoint epoch | Test accuracy | Test macro-F1 |
  |---|---|---|---|
  | Internal | 47 | 0.8693 | 0.5565 |
  | External | 36 | 0.8627 | 0.5244 |

  Internal leads on both metrics, with a larger gap on macro-F1 (+0.032)
  than accuracy (+0.007) — suggestive of a rare-class advantage, not yet
  investigated further. Full artifacts (confusion matrix CSV,
  top-10 confused pairs, raw JSON) in
  `outputs/internal/eval/best_model_2026-07-20T10-37-00.../` and
  `outputs/external/eval/best_model_2026-07-20T10-37-42.../`.
- Everything structural from before still holds (`card_parser.py`,
  `dataset.py`, `esm2_wrapper.py`, `soft_prompt.py`, `preprocess_card.py`,
  `run_training.py`, `load_config`, all five configs, the label-leakage fix
  itself).

## What's In Progress

- Nothing actively running. Next action is deciding what, if anything, to
  dig into on the macro-F1 gap (e.g. per-class confusion breakdown) before
  treating the internal-vs-external comparison as final for the poster.

## What's Not Started

1. **Poster write-up** — no longer blocked on retraining; the table above is
   the first trustworthy internal-vs-external comparison since the
   label-leakage fix.
2. **`src/data/prodigal_runner.py`** — still an empty stub, deferred to V2
   (Aidan, 2026-07-09). Not needed for V1 since training/eval only consume
   CARD's pre-translated amino acid FASTA.
3. Reflecting the V2 scope change (TA loci pulled forward from V3, see below)
   in CLAUDE.md's Versioned Roadmap section — still pending, Aidan's to do.

## Open Questions / Blockers

- **Stale hardcoded test:** `tests/test_config.py::TestLoadConfig::test_hyperparameters_are_v1_defaults`
  asserts `training.batch_size == 32` for both GPU configs, but both were
  intentionally changed to `batch_size: 24` on 2026-07-17 (uncommitted as of
  this session — `git status` still shows both configs modified). The
  change is a hardware fix, not a hyperparameter tweak: training moved to
  `sjsu` (RTX 3090), and the 3090s have a lower memory ceiling than
  originally assumed — batch 24 is the confirmed practical max there, vs.
  batch 32 on `spark-833c` (DGX Spark). This is the one failure in an
  otherwise 132/132-passing smoke suite. Since the config edit was applied
  identically to both ablation configs, the A/B comparison itself isn't
  desynced — just this one test's hardcoded expectation. Needs either the
  test updated to 24 or the configs reverted, and the config changes
  committed either way.
- **Is the macro-F1 gap (internal vs. external) a real effect or noise?**
  Both trained under identical hyperparameters except `injection_mode`, so
  it's a clean A/B, but only a single seed/run per arm — no variance
  estimate yet. Worth a second seed before leaning on this for the poster's
  headline claim, time permitting.
- **V2 scope update (2026-07-17, Aidan's decision, not yet in CLAUDE.md):**
  TA loci (TADB 3.0) pulled forward from V3 into V2, alongside RefSeq — V2
  will map CARD metadata + TA loci onto RefSeq.
- `docs/reviews/2026-07-05-classifier-loss-review.md` — remaining deferred
  items are low-severity doc/style nitpicks; unchanged.

## Recent Changes

1. **GPU driver blocker on `sjsu` resolved** — CUDA confirmed working
   (verified 2026-07-20), unblocking the retraining below.
2. **Both ablation configs retrained end-to-end** (50 epochs each) against
   the fixed, non-leaky `ClassifierHead` — the first checkpoints trained
   entirely post-fix.
3. **Holdout evaluation run for both ablations** via `src.eval.evaluate`:
   internal 0.8693 accuracy / 0.5565 macro-F1 (epoch 47) vs. external 0.8627
   accuracy / 0.5244 macro-F1 (epoch 36). This is the first trustworthy
   internal-vs-external number since the label-leakage fix.
4. Flagged a stale hardcoded test (`test_config.py` still expects
   `batch_size: 32`) that now fails against the intentionally-changed
   `batch_size: 24` configs — changed because `sjsu`'s RTX 3090s cap out at
   batch 24, lower than the batch 32 originally assumed from `spark-833c`.
