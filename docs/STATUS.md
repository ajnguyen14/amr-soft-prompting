# AMR Soft Prompting — Project Status
_Last updated: 2026-07-09 14:37_

## Current Version

**V1 — Core Pipeline**, essentially complete on the code side. Data, model,
config, training, eval, and scripts layers are all done and tested (135/135
tests passing). Pipeline has been validated end-to-end on the real CARD
dataset with the 150M model. **Still blocked on the GPU server's driver
setup** for the actual 650M training runs — see Open Questions / Blockers.
Aidan is moving execution to a different (CUDA-capable) server; the repo is
now in a state where a fresh clone can run `preprocess_card.py` then
`run_training.py` end to end without manual steps. `prodigal_runner.py`
(nucleotide → AA translation) was explicitly deferred to V2, despite being
listed under V1 in CLAUDE.md's roadmap — see What's Not Started.

## Completion

**~98% of V1.** Breakdown:
- Data layer: 100% (card_parser, AMRDataset incl. `split_dataset`, both tested)
- Model layer: 100% (esm2_wrapper incl. `output_dim`, soft_prompt, classifier
  all done and tested)
- Config layer: 100% (`base.yaml` + all four per-environment overrides +
  `load_config` merge utility, all done and tested; `gpu_server_internal.yaml`/
  `gpu_server_external.yaml` confirmed to already have separate `output_dir`s,
  the one intentional exception to the "mirror every field" rule; `local.yaml`
  now populated per the local WSL2 tier, no longer an empty stub)
- Training layer: 100% (`loss.py` + `train.py`, both done and tested;
  `train.py` now loads the preprocessed split artifact from
  `preprocess_card.py` when present instead of re-parsing raw CARD every run)
- Eval layer: 100% (`metrics.py` + `evaluate.py`, both done and tested —
  confusion matrices + per-class breakdown for `resistance_mechanism`/
  `drug_class`, aggregate + top-confused-pairs for the 398-class
  `amr_gene_family`, self-contained JSON reproducibility artifact)
- Scripts: 100% (`preprocess_card.py` + `run_training.py`, both new this
  session, both done and manually verified against real CARD data)
- Tests: ~98% of planned coverage (data + split + dataset + esm2_wrapper +
  soft_prompt + classifier + config loader + metrics + train.py + evaluate.py,
  both injection modes) — 135/135 passing. No dedicated smoke test added for
  the two new scripts (thin CLI wrappers around already-tested library code);
  verified manually instead (see Recent Changes).

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
  Entry point: `python -m src.training.train --config configs/gpu_server_internal.yaml`
  (now also reachable via `scripts/run_training.py`, see below).
  `build_dataloaders` now loads the split artifact `preprocess_card.py` writes
  (via new `load_split_artifact` in `dataset.py`) when present, falling back
  to re-parsing raw CARD only if it isn't — closes the gap noted in the prior
  session where the two scripts didn't actually share the split.
- `src/eval/evaluate.py` — CLI: `python -m src.eval.evaluate --config <path>
  --checkpoint <path>`. Rebuilds the exact test split/model via `train.py`'s
  own builders, reuses `metrics.py` for aggregate numbers. Full confusion
  matrix + per-class precision/recall/F1 (rendered to PNG) for
  `resistance_mechanism` and `drug_class`; for the 398-class `amr_gene_family`,
  aggregate accuracy + macro-F1 + top-10 confused-pairs table only, with the
  raw 398×398 matrix dumped to CSV (never rendered — not readable at that
  size). Every run writes a self-contained `evaluation_results.json`
  (checkpoint path, full config, timestamp, all metrics) as the reproducibility
  artifact. No wandb logging — a point-in-time report, not a run to log.
- `scripts/preprocess_card.py` — CLI: `python
  scripts/preprocess_card.py --config <path>`. Loads raw CARD files per
  config, runs `load_card_dataset` + `get_label_vocabularies` +
  `split_dataset` (same `SEED=42` as `train.py`), and pickles
  `{splits, label_vocabularies}` to `<output_dir>/card_splits.pkl` via the new
  shared `save_split_artifact` helper in `dataset.py`. Prints per-split record
  counts and vocab sizes on completion. Manually verified against real CARD
  data via `configs/cpu_server.yaml`: reproduced the known 4839/601/612 split
  and 38/10/398 vocab sizes exactly.
- `scripts/run_training.py` — thin CLI wrapper: `python
  scripts/run_training.py --config <path>` calls `load_config` then
  `src.training.train.train()` directly, no new logic. Verified via `--help`
  and import resolution; a full training run was not exercised here since
  gradient-update training belongs on the GPU server per CLAUDE.md's
  escalation rules, not the CPU server this was written on.
  `train.py`'s `build_dataloaders` now consumes `preprocess_card.py`'s
  pickled split artifact directly via `load_split_artifact` when it exists at
  the configured `output_dir`, closing the gap flagged in the prior session
  (previously it silently re-parsed and re-split raw CARD every run instead).
- `src/utils/config.py` (`load_config`) — deep-merges an environment config
  over `configs/base.yaml`.
- `configs/base.yaml`, `configs/cpu_server.yaml`, `configs/gpu_server_internal.yaml`,
  `configs/gpu_server_external.yaml`, `configs/local.yaml` — all five real and
  validated. `local.yaml` was previously an empty stub (silently fell back to
  `base.yaml` alone and crashed any script reading model/paths config); now
  populated per CLAUDE.md's local WSL2 tier (8M model, CPU, small batch).
- `requirements.txt` — `wandb` pinned to `0.28.0`, matching the installed
  version.
- `tests/` — 135/135 passing (data pipeline + dataset/split + esm2_wrapper +
  soft_prompt + classifier + config loader + metrics + train.py + evaluate.py,
  both injection modes throughout).
- **150M pipeline validation on real CARD data** (prior session, ad hoc) — ran
  the full pipeline (`AMRDataset → ESM2Wrapper(150M) → SoftPromptModule →
  ClassifierHead → AMRLoss`) over the entire real test split (612 records) on
  CPU, both injection modes, inference only. Confirmed: 0 trainable ESM-2
  params in both modes, all finite losses, no shape errors.
- `docs/sessions/` — logs for all sessions to date.
- `docs/reviews/` — `/code-review` findings logged per session.

## What's In Progress

Nothing actively mid-implementation. All V1 library code (`src/`) and both
entry-point scripts are complete and tested. Remaining work is the actual GPU
training runs, blocked on the driver fix (see below).

## What's Not Started

1. **`src/data/prodigal_runner.py`** — nucleotide → AA translation. Still an
   empty stub. CLAUDE.md's roadmap lists this under V1, but Aidan explicitly
   deferred it to V2 (2026-07-09): V1 training/eval only ever consumes CARD's
   pre-translated amino acid FASTA, so Prodigal isn't actually needed to reach
   full V1 functional completion — it only matters for running inference on
   raw, unannotated nucleotide genomes rather than CARD's curated protein set.
   Pick this up first when V2 work starts.

## Open Questions / Blockers

- **GPU server's CUDA stack is broken — blocks the real 650M training runs.**
  Hardware is present and correct (2× RTX 3090, 24GB VRAM each, confirmed via
  `lspci`/`/proc/driver/nvidia/gpus/`), but `torch.cuda.is_available()` returns
  `False` with error 804 ("forward compatibility was attempted on non
  supported HW"). Root cause: two NVIDIA driver package sets are installed
  side by side (570.211.01 and 580.159.03); the loaded kernel module is still
  570.211.01 as of 2026-07-09, but PyTorch's CUDA runtime needs the 580-series
  driver. `nvidia-smi` itself still isn't installed (`nvidia-compute-utils-570`
  shows as removed/`rc` in `dpkg`) — re-confirmed unchanged from the original
  2026-07-06 diagnosis. Fixing this needs a driver reconciliation
  (reinstall matching kernel module + userspace libs, likely a reboot) —
  system-level, out of scope for in-repo work. **Aidan is moving to a
  different server to work around this rather than waiting on the fix.**
- **No real GPU training run has happened yet** (blocked by the above, now
  sidestepped by the server move). Once on CUDA-capable hardware:
  `python scripts/preprocess_card.py --config <env>.yaml` then `python
  scripts/run_training.py --config configs/gpu_server_internal.yaml` (and
  `..._external.yaml`) are ready to run — `evaluate.py` exists to make sense
  of the results.
- **`docs/reviews/2026-07-05-classifier-loss-review.md`** — remaining deferred
  items are low-severity doc/style nitpicks; see that file for the list.

## Recent Changes

1. **`prodigal_runner.py` explicitly deferred to V2** — Aidan's call
   (2026-07-09), even though CLAUDE.md's roadmap lists Prodigal gene-calling
   under V1. Not needed for CARD-FASTA training/eval; only matters for
   real-genome inference. Remains an empty stub until V2 starts.
2. **`train.py` wired to consume `preprocess_card.py`'s preprocessed split**
   — added `save_split_artifact`/`load_split_artifact` to `dataset.py`,
   shared by both scripts; `build_dataloaders` now loads the pickled split
   when present instead of re-parsing and re-splitting raw CARD on every
   run. Closes the gap flagged in the prior session.
3. **`configs/local.yaml` populated** — was an empty file, so
   `--config configs/local.yaml` silently fell back to `base.yaml` alone and
   crashed any script reading model/paths config. Filled in per CLAUDE.md's
   local WSL2 tier (8M model, CPU, small batch).
4. **`requirements.txt`: `wandb` pinned to `0.28.0`**, the installed version.
5. **`scripts/preprocess_card.py` implemented** — the CLAUDE.md-mandated
   single-entry-point preprocessing script. Parses raw CARD files, splits,
   and pickles the result plus label vocabularies to `output_dir`; prints a
   record-count/vocab-size summary. Verified against real CARD data.
6. **`scripts/run_training.py` implemented** — thin wrapper making the
   CLAUDE.md-documented `python scripts/run_training.py --config ...`
   invocation actually work; delegates entirely to `src/training/train.py`.
7. **Fixed a script-execution import gap surfaced while building the above:**
   plain `python scripts/<name>.py` execution puts only `scripts/` on
   `sys.path`, not the repo root, so `from src... import ...` failed until
   both scripts added the same `sys.path.insert(parent-of-scripts)` pattern
   `conftest.py` already used for pytest.
8. **Confirmed `gpu_server_internal.yaml`/`gpu_server_external.yaml` already
   have separate `output_dir`s** (`outputs/internal/`, `outputs/external/`)
   from a prior session's fix — no change needed, correctly the one
   intentional exception to the config-mirroring rule.
9. **Re-verified the GPU server's CUDA breakage is unchanged** (driver
   mismatch, missing `nvidia-smi`) as of 2026-07-09 — prompted the decision
   to move training execution to a different server rather than continue
   waiting on the system-level fix.
10. **`src/eval/evaluate.py` implemented and tested** (prior session) — full
    V1 holdout evaluation, confusion matrices, JSON reproducibility artifact.
