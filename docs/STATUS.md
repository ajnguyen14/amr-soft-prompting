# AMR Soft Prompting — Project Status
_Last updated: 2026-07-06 00:00_

## Current Version

**V1 — Core Pipeline**, late-stage. Data, model, config, training, and eval
layers are all complete and tested (135/135 tests passing). Pipeline has now
been validated end-to-end on the real CARD dataset with the 150M model.
**Blocked on the GPU server's driver setup** for the actual 650M training
runs — see Open Questions / Blockers.

## Completion

**~95% of V1.** Breakdown:
- Data layer: 100% (card_parser, AMRDataset incl. `split_dataset`, both tested)
- Model layer: 100% (esm2_wrapper incl. `output_dim`, soft_prompt, classifier
  all done and tested)
- Config layer: 100% (`base.yaml` + all four per-environment overrides +
  `load_config` merge utility, all done and tested; `cpu_server.yaml` was
  actually an empty stub until this session despite being marked done
  previously — now filled in and validated against real data)
- Training layer: 100% (`loss.py` + `train.py`, both done and tested)
- Eval layer: 100% (`metrics.py` + `evaluate.py`, both done and tested —
  confusion matrices + per-class breakdown for `resistance_mechanism`/
  `drug_class`, aggregate + top-confused-pairs for the 398-class
  `amr_gene_family`, self-contained JSON reproducibility artifact)
- Scripts: 0% (`preprocess_card.py`, `run_training.py` still empty stubs)
- Tests: ~98% of planned coverage (data + split + dataset + esm2_wrapper +
  soft_prompt + classifier + config loader + metrics + train.py + evaluate.py,
  both injection modes) — 135/135 passing

## What's Working

- `src/data/card_parser.py` — CARD FASTA + ARO index → `CARDRecord` objects;
  label vocabularies for drug class (38), resistance mechanism (10), AMR gene
  family (398). Tested against real CARD v4.0.1 (6052 records).
- `src/data/dataset.py` (`split_dataset`) — train/val/test split on ARO
  accession, stratified by resistance_mechanism, 80/10/10, deterministic via a
  local `random.Random(seed)`. Verified against real CARD: 4839/601/612
  train/val/test, zero cross-split accession overlap.
- `src/data/dataset.py` (`AMRDataset`) — multi-hot float32 for drug class
  (BCEWithLogitsLoss), scalar long for mechanism/family (CrossEntropyLoss).
- `src/models/esm2_wrapper.py` (`ESM2Wrapper`) — frozen ESM-2, both injection
  modes, correct residue-only mean pooling, `output_dim(num_prompt_tokens)` as
  single source of truth for the mode-dependent output width.
- `src/models/soft_prompt.py` (`SoftPromptModule`) — resistance mechanism via
  `nn.Embedding`, drug class via multi-hot × embedding matrix, 2 soft prompt
  tokens of shape `(B, 2, embed_dim)`, all trainable.
- `src/models/classifier.py` (`ClassifierHead`) — shared trunk feeding three
  independent heads (`drug_class`, `resistance_mechanism`, `amr_gene_family`).
- `src/training/loss.py` (`AMRLoss`) — BCEWithLogitsLoss + 2×CrossEntropyLoss,
  weighted sum (default 1.0 each), sourced from `configs/base.yaml`.
- `src/eval/metrics.py` (`compute_metrics`) — shared per-task metric helper
  used by both `train.py`'s per-epoch logging and `evaluate.py`'s holdout eval.
- `src/training/train.py` — full V1 training loop: builds train/val/test
  loaders (vocab from the full dataset before splitting), trains with Adam,
  validates every epoch, logs to wandb, checkpoints only on best val loss.
  Entry point: `python -m src.training.train --config configs/gpu_server_internal.yaml`.
- `src/eval/evaluate.py` (new this session) — CLI: `python -m src.eval.evaluate
  --config <path> --checkpoint <path>`. Rebuilds the exact test split/model via
  `train.py`'s own builders, reuses `metrics.py` for aggregate numbers. Full
  confusion matrix + per-class precision/recall/F1 (rendered to PNG) for
  `resistance_mechanism` and `drug_class`; for the 398-class `amr_gene_family`,
  aggregate accuracy + macro-F1 + top-10 confused-pairs table only, with the
  raw 398×398 matrix dumped to CSV (never rendered — not readable at that
  size). Every run writes a self-contained `evaluation_results.json`
  (checkpoint path, full config, timestamp, all metrics) as the reproducibility
  artifact. No wandb logging — a point-in-time report, not a run to log.
- `src/utils/config.py` (`load_config`) — deep-merges an environment config
  over `configs/base.yaml`.
- `configs/base.yaml`, `configs/cpu_server.yaml`, `configs/gpu_server_internal.yaml`,
  `configs/gpu_server_external.yaml` — all four now real and validated (see
  Recent Changes for this session's fixes). `configs/local.yaml` is still an
  empty stub (not yet needed — no local-only work has required it).
- `tests/` — 135/135 passing (data pipeline + dataset/split + esm2_wrapper +
  soft_prompt + classifier + config loader + metrics + train.py + evaluate.py,
  both injection modes throughout).
- **150M pipeline validation on real CARD data** (this session, ad hoc, not
  committed as a script) — ran the full pipeline
  (`AMRDataset → ESM2Wrapper(150M) → SoftPromptModule → ClassifierHead → AMRLoss`)
  over the entire real test split (612 records) on CPU, both injection modes,
  inference only. Confirmed: 0 trainable ESM-2 params in both modes, all
  finite losses, no shape errors. Accuracy numbers are near-random as
  expected — `soft_prompt`/`classifier` are untrained, this was a plumbing
  check only.
- `docs/sessions/` — logs for all sessions to date.
- `docs/reviews/` — `/code-review` findings logged per session.

## What's In Progress

Nothing actively mid-implementation. All V1 library code (`src/`) is complete
and tested. Remaining work is the two entry-point scripts and the actual GPU
training runs, both blocked/pending (see below).

## What's Not Started

In intended build order:

1. **`scripts/preprocess_card.py`** — one-time preprocessing entry point
2. **`scripts/run_training.py`** — thin CLI wrapper; `src/training/train.py`
   already has a working `python -m src.training.train --config ...` entry
   point, so this script mostly needs to exist for the CLAUDE.md-documented
   `scripts/run_training.py --config ...` invocation path
3. **`src/data/prodigal_runner.py`** — nucleotide → AA translation (needed for
   real-genome inference, not required for CARD FASTA training)

## Open Questions / Blockers

- **GPU server's CUDA stack is broken — blocks the real 650M training runs.**
  Hardware is present and correct (2× RTX 3090, 24GB VRAM each, confirmed via
  `lspci`/`/proc/driver/nvidia/gpus/`), but `torch.cuda.is_available()` returns
  `False` with error 804 ("forward compatibility was attempted on non
  supported HW"). Root cause: two NVIDIA driver package sets are installed
  side by side (570.211.01 and 580.159.03); the loaded kernel module is
  570.211.01 but PyTorch's CUDA runtime is 13.0, which needs the 580-series
  driver. `nvidia-smi` itself isn't installed (`nvidia-compute-utils-570`
  shows as removed/`rc` in `dpkg`). Fixing this needs a driver reconciliation
  (reinstall matching kernel module + userspace libs, likely a reboot) —
  system-level, out of scope for in-repo work. Aidan is looping in Prof.
  Andreopolous to get this fixed. Until then: no gradient-update training and
  no 650M-model inference on this box; 8M/150M CPU-mode work is unaffected.
- **No real GPU training run has happened yet** (blocked by the above). Once
  unblocked: `python -m src.training.train --config
  configs/gpu_server_internal.yaml` and `..._external.yaml` are ready to run —
  `evaluate.py` exists to make sense of the results.
- **`docs/reviews/2026-07-05-classifier-loss-review.md`** — remaining deferred
  items are low-severity doc/style nitpicks; see that file for the list.

## Recent Changes

1. **`src/eval/evaluate.py` implemented and tested** — full V1 holdout
   evaluation. Confusion matrices + per-class breakdown for
   `resistance_mechanism`/`drug_class`; aggregate + macro-F1 + top-confused
   pairs for `amr_gene_family` (398 classes, too large to render); JSON
   reproducibility artifact per run. Added `matplotlib`/`seaborn` to
   `requirements.txt` for the figures. 7 new tests in `tests/test_evaluate.py`.
2. **Fixed two real config bugs found while running the 150M CPU validation:**
   - `configs/cpu_server.yaml` was an empty stub (marked "done" in a prior
     status update in error) — filled in with real CARD paths,
     `facebook/esm2_t30_150M_UR50D`, `device: cpu`, `batch_size: 8` per
     CLAUDE.md's CPU-server inference cap.
   - `gpu_server_internal.yaml`/`gpu_server_external.yaml`'s `esm2_variant`
     was missing the `facebook/` HF org prefix (`"esm2_t33_650M_UR50D"`
     instead of `"facebook/esm2_t33_650M_UR50D"`) — would have failed to
     resolve on the Hub and broken the real GPU training run before this was
     caught.
   - Gave the two ablation configs separate `output_dir`s
     (`outputs/internal/`, `outputs/external/`) so they no longer silently
     overwrite each other's checkpoint; updated `test_config.py` accordingly.
3. **Ran a full pipeline validation on real CARD data with the 150M model**
   (both injection modes, CPU-only, inference-only, 612 real test records) —
   confirmed no shape errors and ESM-2 stays frozen; this is what surfaced the
   two config bugs above.
4. **Diagnosed the GPU server's CUDA breakage** — driver version mismatch
   (570 kernel module vs. 580-expecting CUDA 13.0 runtime), missing
   `nvidia-smi`. Documented as the current top blocker; professor is being
   looped in to fix it at the system level.
5. **`train.py` implemented and tested** (prior session) — the full V1
   training loop, checkpointing on best val loss, wandb logging of all loss
   terms and accuracy/F1 metrics.
6. **`src/eval/metrics.py` added** (prior session) — shared `compute_metrics`
   helper so `train.py` and `evaluate.py` never compute metrics two different
   ways.
