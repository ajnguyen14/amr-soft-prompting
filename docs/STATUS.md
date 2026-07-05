# AMR Soft Prompting — Project Status
_Last updated: 2026-07-05 00:00_

## Current Version

**V1 — Core Pipeline**, mid-implementation. Data layer, model layer, config
system, and now the training loop are all complete and tested. Next:
`src/eval/evaluate.py` (fuller holdout eval), then the two entry-point scripts.

## Completion

**~95% of V1.** Breakdown:
- Data layer: 100% (card_parser, AMRDataset incl. `split_dataset`, both tested)
- Model layer: 100% (esm2_wrapper incl. `output_dim`, soft_prompt, classifier
  all done and tested)
- Config layer: 100% (`base.yaml` + per-environment overrides + `load_config`
  merge utility, all done and tested)
- Training layer: 100% (`loss.py` + `train.py`, both done and tested)
- Eval layer: ~20% (`src/eval/metrics.py` — shared per-epoch metric helper —
  done and tested; `src/eval/evaluate.py`'s fuller per-class F1/confusion
  matrix holdout eval still an empty stub)
- Scripts: 0% (preprocess_card.py, run_training.py empty stubs)
- Tests: ~98% of planned coverage (data + split + dataset + esm2_wrapper +
  soft_prompt + classifier + config loader + metrics + full training loop,
  both injection modes)

## What's Working

- `src/data/card_parser.py` — CARD FASTA + ARO index → `CARDRecord` objects;
  label vocabularies for drug class, resistance mechanism, AMR gene family.
  Tested against real CARD v4.0.1 (6052 records).
- `src/data/dataset.py` (`split_dataset`) — train/val/test split on ARO
  accession (not sequence, to prevent multi-sequence-per-accession leakage),
  stratified by resistance_mechanism, 80/10/10 default ratio, deterministic via
  a local `random.Random(seed)` instance (default seed 42) that never touches
  global RNG state. Verified against the real CARD dataset: full coverage, zero
  cross-split accession overlap, ~80/10/10 actual ratio.
- `src/data/dataset.py` (`AMRDataset`) — multi-hot float32 for drug class
  (BCEWithLogitsLoss), scalar long for mechanism/family (CrossEntropyLoss).
  Default DataLoader collation works without a custom `collate_fn`.
- `src/models/esm2_wrapper.py` (`ESM2Wrapper`) — frozen ESM-2 with both injection
  modes. Correct residue-only mean pooling. Gradient test confirms `soft_prompt_vectors`
  receive gradients while ESM-2 parameters stay frozen. `output_dim(num_prompt_tokens)`
  is now the single source of truth for the mode-dependent output width formula
  (`embed_dim` internal, `embed_dim + N*embed_dim` external) — `train.py` and
  `test_forward_pass.py` both call it instead of re-deriving the formula.
- `src/models/soft_prompt.py` (`SoftPromptModule`) — encodes resistance mechanism
  (`nn.Embedding` lookup) and drug class (multi-hot vector times the embedding weight
  matrix, mathematically equivalent to sum-pooling over active class embeddings) into
  2 soft prompt tokens of shape `(B, 2, embed_dim)`. All parameters trainable.
  `NUM_PROMPT_TOKENS = 2` class constant is the single source of truth for the
  token count (paired with `ESM2Wrapper.output_dim`, above).
- `src/models/classifier.py` (`ClassifierHead`) — shared trunk
  (`Linear(input_dim, hidden_dim)` → `ReLU` → `Dropout`) feeding three independent
  heads for `drug_class`, `resistance_mechanism`, `amr_gene_family`. `input_dim` is
  a constructor arg, so the same class serves both `ESM2Wrapper` injection modes.
  Returns a dict of per-task logit tensors.
- `src/training/loss.py` (`AMRLoss`) — BCEWithLogitsLoss (drug_class, summed then
  divided by batch_size) + CrossEntropyLoss (resistance_mechanism,
  amr_gene_family), weighted sum with constructor-arg weights (default 1.0 each,
  matching the locked-in α=β=γ=1 decision, now actually sourced from
  `configs/base.yaml`'s `loss:` section via `train.py`). Returns a dict of
  per-task losses plus `total` for logging.
- `src/eval/metrics.py` (`compute_metrics`) — shared per-task metric helper:
  accuracy for the two single-label heads, micro-F1 (thresholded at 0.5) for
  multi-label drug_class. Used by `train.py`'s per-epoch logging; will also
  back `evaluate.py`'s fuller holdout eval so the two never compute metrics two
  different ways.
- `src/training/train.py` — the full V1 training loop. Builds train/val/test
  `DataLoader`s (label vocab built from the *full* dataset before splitting, so
  val/test never hit an out-of-vocab label), constructs the frozen ESM-2 +
  soft prompt + classifier + loss from config, trains with `Adam`, validates
  every epoch, logs total loss, all three individual loss terms, and the
  accuracy/F1 metrics to wandb every epoch, and checkpoints only on a new best
  total validation loss (not every epoch) — only `soft_prompt`/`classifier`
  state dicts are saved, since ESM-2 is frozen and fully determined by config.
  Entry point: `python -m src.training.train --config configs/gpu_server_internal.yaml`.
- `src/utils/config.py` (`load_config`) — loads an environment config file and
  deep-merges it over `configs/base.yaml` (nested sections merge key-by-key, not
  wholesale replacement, so an override only needs to specify what it actually
  changes). Regression-tested against a real PyYAML gotcha: unquoted `1e-4`
  (no decimal point) parses as a string, not a float — configs must write
  `1.0e-4` or `0.0001`.
- `configs/base.yaml` — shared defaults: `model.freeze_esm2` (documentation
  only — `ESM2Wrapper` always freezes unconditionally, this field isn't read
  by any code), `training` (`batch_size: 32`, `learning_rate: 1.0e-4`,
  `epochs: 50`, `optimizer: "adam"` — the V1 defaults), `classifier`
  (`hidden_dim: 512`, `dropout: 0.1`), `loss` (equal weights, all `1.0`),
  `logging` (`wandb_project: "amr-soft-prompting"`, `wandb_run_name: null`).
- `configs/gpu_server_internal.yaml` / `configs/gpu_server_external.yaml` — the
  two ablation run configs (650M model, cuda), identical in every field except
  `model.injection_mode`. The old single `configs/gpu_server.yaml` has been
  deleted now that both replacements are in place.
- `tests/test_forward_pass.py` — full pipeline integration test: ESM2Wrapper →
  SoftPromptModule → ClassifierHead → AMRLoss, 8M model, batch=2, parametrized
  over both injection modes.
- `tests/test_train.py` — 10 tests covering `build_dataloaders_from_records`,
  `build_optimizer`, `run_epoch` (train vs. eval mode; confirms training
  actually updates soft_prompt/classifier weights and eval mode doesn't), and a
  full `train()` integration test (both injection modes) against a synthetic
  20-sequence CARD-like FASTA+TSV fixture on disk — the first test to exercise
  the real `load_card_dataset` → `AMRDataset` → `DataLoader` path end-to-end,
  closing a gap `test_forward_pass.py` had left open. wandb disabled via
  `WANDB_MODE=disabled` so tests never touch the network.
- `tests/` — 128/128 passing (22 data pipeline + 28 dataset incl. split + 35
  esm2_wrapper incl. `output_dim` + 6 soft_prompt + 9 classifier + 2 end-to-end
  forward/backward + 11 config loader + 5 metrics + 10 train.py).
- `docs/sessions/` — logs for all sessions to date.
- `docs/reviews/` — `/code-review` findings logged per session, with
  fixed/deferred status tracked so open items aren't lost.

## What's In Progress

Nothing actively mid-implementation. `train.py` complete; `evaluate.py` is next.

## What's Not Started

In intended build order:

1. **`src/eval/evaluate.py`** — fuller per-class F1, confusion matrix, CARD
   holdout eval. Should build on `src/eval/metrics.py` rather than
   recomputing accuracy/F1 a second way.
2. **`scripts/preprocess_card.py`** — one-time preprocessing entry point
3. **`scripts/run_training.py`** — thin CLI wrapper; `src/training/train.py`
   already has a working `python -m src.training.train --config ...` entry
   point, so this script mostly needs to exist for the CLAUDE.md-documented
   `scripts/run_training.py --config ...` invocation path
4. **`src/data/prodigal_runner.py`** — nucleotide → AA translation (needed for
   real-genome inference, not required for CARD FASTA training)

## Open Questions / Blockers

- **No real GPU training run has happened yet.** Everything is validated on
  CPU with the 8M model and synthetic/tiny data. The actual 650M runs on real
  CARD data (`configs/gpu_server_internal.yaml` / `gpu_server_external.yaml`)
  still need to happen on the GPU server — this is the natural next real
  milestone once `evaluate.py` exists to make sense of the results.
- **`docs/reviews/2026-07-05-classifier-loss-review.md`** — most findings now
  fixed this session (loss-weights wiring, `input_dim` duplication, the
  `AMRDataset` test-coverage gap). Remaining deferred items are all low-severity
  doc/style nitpicks; see that file for the current list.

## Recent Changes

1. **`train.py` implemented and tested** — the full V1 training loop:
   `build_dataloaders_from_records`/`build_dataloaders`, `build_models`,
   `build_optimizer`, `run_epoch`, `train`. Checkpoints on best total val loss
   only, logs all three individual loss terms plus total and accuracy/F1
   metrics to wandb every epoch, per this session's explicit design decisions.
   10 new tests in `tests/test_train.py`, including a full end-to-end run
   against a synthetic on-disk CARD fixture for both injection modes.
2. **`src/eval/metrics.py` added** — shared `compute_metrics` helper (accuracy
   for single-label tasks, micro-F1 for multi-label drug_class) so `train.py`'s
   per-epoch logging and the future `evaluate.py` holdout eval compute metrics
   the same way once, not twice. 5 new tests.
3. **Fixed the duplicated `input_dim`/N formula** (deferred finding from the
   2026-07-05 code review) — added `ESM2Wrapper.output_dim(num_prompt_tokens)`
   and `SoftPromptModule.NUM_PROMPT_TOKENS`; `train.py` and
   `test_forward_pass.py` now call the shared method instead of re-deriving
   `embed_dim + N*embed_dim` a fourth time.
4. **Added `scikit-learn` as a real dependency** — it was listed in CLAUDE.md's
   key packages but never actually pinned in `requirements.txt` or installed;
   needed for `compute_metrics`'s F1 computation. Installed
   (`scikit-learn==1.9.0`) and added to `requirements.txt`.
5. **Config system completed** (prior sub-session, same day) —
   `configs/base.yaml`'s `model:`/`training:`/`logging:` sections; two ablation
   configs (`gpu_server_internal.yaml`/`gpu_server_external.yaml`) replacing
   the old ambiguous `gpu_server.yaml`; `src/utils/config.py` (`load_config`)
   deep-merge utility; `CLAUDE.md` updated throughout.
6. **`split_dataset` implemented and tested** (prior sub-session, same day) —
   train/val/test split on ARO accession, stratified by resistance_mechanism,
   80/10/10 default, deterministic local RNG.
