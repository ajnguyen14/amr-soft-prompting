# AMR Soft Prompting — Project Status
_Last updated: 2026-07-05 00:00_

## Current Version

**V1 — Core Pipeline**, mid-implementation. Data layer, model layer, loss, and
config system all complete. Next: `src/training/train.py`.

## Completion

**~85% of V1.** Breakdown:
- Data layer: 100% (card_parser, AMRDataset incl. `split_dataset`, both tested)
- Model layer: 100% (esm2_wrapper, soft_prompt, classifier all done and tested)
- Config layer: 100% (`base.yaml` + per-environment overrides + `load_config`
  merge utility, all done and tested)
- Training layer: ~50% (loss.py done and integration-tested; train.py empty stub)
- Eval layer: 0% (evaluate.py empty stub)
- Scripts: 0% (preprocess_card.py, run_training.py empty stubs)
- Tests: ~95% of planned coverage (data + split + dataset + esm2_wrapper +
  soft_prompt + classifier + config loader + end-to-end forward/backward pass,
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
  receive gradients while ESM-2 parameters stay frozen.
- `src/models/soft_prompt.py` (`SoftPromptModule`) — encodes resistance mechanism
  (`nn.Embedding` lookup) and drug class (multi-hot vector times the embedding weight
  matrix, mathematically equivalent to sum-pooling over active class embeddings) into
  2 soft prompt tokens of shape `(B, 2, embed_dim)`. All parameters trainable.
- `src/models/classifier.py` (`ClassifierHead`) — shared trunk
  (`Linear(input_dim, hidden_dim)` → `ReLU` → `Dropout`) feeding three independent
  heads for `drug_class`, `resistance_mechanism`, `amr_gene_family`. `input_dim` is
  a constructor arg (mode-dependent: `embed_dim` for internal injection,
  `embed_dim + N*embed_dim` for external), so the same class serves both
  `ESM2Wrapper` injection modes. Returns a dict of per-task logit tensors.
- `src/training/loss.py` (`AMRLoss`) — BCEWithLogitsLoss (drug_class, summed then
  divided by batch_size) + CrossEntropyLoss (resistance_mechanism,
  amr_gene_family), weighted sum with constructor-arg weights (default 1.0 each,
  matching the locked-in α=β=γ=1 decision). Returns a dict of per-task losses
  plus `total` for logging. Resolves the `drug_class_labels` → `drug_class`
  key-name mismatch between `AMRDataset` and `ClassifierHead` internally.
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
  (`hidden_dim: 512`, `dropout: 0.1`), `loss` (equal weights, all `1.0`).
- `configs/gpu_server_internal.yaml` / `configs/gpu_server_external.yaml` — the
  two ablation run configs (650M model, cuda), identical in every field except
  `model.injection_mode`. Replace the old single `configs/gpu_server.yaml`
  (which only had `paths:` and no model/training sections) — that file has been
  deleted (`git rm`) now that both replacements are in place.
- `tests/test_forward_pass.py` — full pipeline integration test: ESM2Wrapper →
  SoftPromptModule → ClassifierHead → AMRLoss, 8M model, batch=2, parametrized
  over both injection modes. Confirms forward pass produces all four loss keys,
  and backward pass sends gradients to soft prompt + classifier while every
  ESM-2 parameter stays at `None`/zero grad. Seeded before any module
  construction so weight init is reproducible too.
- `tests/` — 110/110 passing (32 data pipeline incl. split + 18 dataset + 32
  esm2_wrapper + 6 soft_prompt + 9 classifier + 2 end-to-end forward/backward +
  11 config loader).
- `docs/sessions/` — logs for all sessions to date.
- `docs/reviews/` — `/code-review` findings logged per session, with
  fixed/deferred status tracked so open items aren't lost.

## What's In Progress

Nothing actively mid-implementation. Config system complete; `train.py` is next.

## What's Not Started

In intended build order:

1. **`src/training/train.py`** — main training loop with wandb logging. All of
   its dependencies now exist: `load_config`, `AMRDataset` + `split_dataset`,
   `ESM2Wrapper`, `SoftPromptModule`, `ClassifierHead`, `AMRLoss`. Remember the
   TODO in `src/training/loss.py`: must construct `AMRLoss(**config["loss"])`
   explicitly rather than relying on its Python defaults.
2. **`src/eval/evaluate.py`** — per-class F1, confusion matrix, CARD holdout eval
3. **`scripts/preprocess_card.py`** — one-time preprocessing entry point
4. **`scripts/run_training.py`** — training entry point (`--config` argument)
5. **`src/data/prodigal_runner.py`** — nucleotide → AA translation (needed for
   real-genome inference, not required for CARD FASTA training)

## Open Questions / Blockers

- **Optimizer is recorded as a string (`"adam"`) in config but not yet
  instantiated anywhere** — `train.py` will need to map this string to
  `torch.optim.Adam` (or whichever optimizer). No AdamW/weight-decay decision
  has been made; `Adam` with no weight decay is the current V1 default.
- **`docs/reviews/2026-07-05-classifier-loss-review.md`** has the full list of
  deferred code-review findings (test-coverage gaps, duplicated `input_dim`
  formula, doc nitpicks) — worth a pass before/around the first real GPU
  training run, none of them urgent before that.

## Recent Changes

1. **Config system completed** — `configs/base.yaml` now has `model:` and
   `training:` sections (V1 defaults: `lr=1e-4, batch_size=32, epochs=50,
   optimizer=adam`); two new ablation-specific configs
   `configs/gpu_server_internal.yaml` / `gpu_server_external.yaml` replace the
   old ambiguous `gpu_server.yaml`, identical except `injection_mode`; new
   `src/utils/config.py` (`load_config`) deep-merges an environment file over
   `base.yaml`. `CLAUDE.md` updated throughout (Configuration System, Project
   Structure, Workflow sections) to reference the two-file convention and
   state explicitly that hyperparameter changes must be applied to both GPU
   configs. 11 new tests in `tests/test_config.py`, including a regression
   guard for a real PyYAML gotcha (unquoted `1e-4` parses as a string, not a
   float). Full suite: 110/110 passing.
2. **`split_dataset` implemented and tested**, then moved to `src/data/dataset.py`
   (`tests/test_dataset.py`) — train/val/test split on ARO accession, stratified
   by resistance_mechanism, 80/10/10 default, deterministic local RNG. This was
   identified as a genuine blocker for `train.py`: the split strategy was
   decided on 2026-07-02 but never implemented until this session.
3. **`/code-review` run on the classifier+loss session** (8 finder angles, 9
   findings survived verification) — logged to
   `docs/reviews/2026-07-05-classifier-loss-review.md`. Two findings fixed
   immediately (BCE/CE reduction mismatch in `AMRLoss`, RNG seed order in
   `test_forward_pass.py`); the rest logged as deferred with reasoning.
4. **`AMRLoss` implemented and tested** (`src/training/loss.py`) — combines
   BCEWithLogitsLoss (drug_class) + 2× CrossEntropyLoss (mechanism, family) into
   a weighted sum. Handles the `drug_class_labels` → `drug_class` key mapping
   internally rather than renaming keys in `AMRDataset` or `ClassifierHead`.
5. **`tests/test_forward_pass.py` implemented** — first true end-to-end test of
   the full V1 model stack, parametrized over both injection modes, verifying
   both forward and backward passes.
6. **`ClassifierHead` implemented and tested** — shared trunk + three task
   heads, dict-shaped output.
