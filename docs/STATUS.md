# AMR Soft Prompting — Project Status
_Last updated: 2026-07-05 00:00_

## Current Version

**V1 — Core Pipeline**, mid-implementation. Data layer complete (including the
train/val/test split, implemented this session). Model layer complete. Loss and
full-pipeline integration test complete. Next: fill in `configs/base.yaml`'s
model/training sections, resolve remaining hyperparameter decisions, then
`train.py`.

## Completion

**~75% of V1.** Breakdown:
- Data layer: 100% (card_parser, AMRDataset incl. `split_dataset`, both tested)
- Model layer: 100% (esm2_wrapper, soft_prompt, classifier all done and tested)
- Training layer: ~50% (loss.py done and integration-tested; train.py empty stub)
- Eval layer: 0% (evaluate.py empty stub)
- Scripts: 0% (preprocess_card.py, run_training.py empty stubs)
- Tests: ~90% of planned coverage (data + split + dataset + esm2_wrapper +
  soft_prompt + classifier + end-to-end forward/backward pass, both injection
  modes)

## What's Working

- `src/data/card_parser.py` — CARD FASTA + ARO index → `CARDRecord` objects;
  label vocabularies for drug class, resistance mechanism, AMR gene family.
  Tested against real CARD v4.0.1 (6052 records).
- `src/data/dataset.py` (`split_dataset`) — train/val/test split on ARO
  accession (not sequence, to prevent multi-sequence-per-accession leakage),
  stratified by resistance_mechanism, 80/10/10 default ratio, deterministic via
  a local `random.Random(seed)` instance (default seed 42) that never touches
  global RNG state. Verified against the real CARD dataset: full coverage, zero
  cross-split accession overlap, ~80/10/10 actual ratio. Lives in `dataset.py`,
  not `card_parser.py` — `card_parser.py` is responsible only for parsing raw
  CARD files into `CARDRecord` objects; split logic operates on already-parsed
  records so it belongs with the rest of the dataset-preparation code.
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
  matching the locked-in α=β=γ=1 decision). The manual batch_size normalization
  on the BCE term matters: BCEWithLogitsLoss's default mean divides by
  batch_size × num_classes while CrossEntropyLoss's mean divides by batch_size
  only, so without this fix equal weights wouldn't give equal gradient
  contribution. Resolves the `drug_class_labels` → `drug_class` key-name
  mismatch between `AMRDataset` and `ClassifierHead` internally, without
  touching either. Returns a dict of per-task losses plus `total` for logging.
- `configs/base.yaml` — has `classifier` (`hidden_dim: 512`, `dropout: 0.1`) and
  `loss` (`weight_drug_class`/`weight_resistance_mechanism`/
  `weight_amr_gene_family`, all `1.0`) sections. Model architecture and training
  sections still empty; the `loss` section isn't read by any code yet (see Open
  Questions).
- `configs/gpu_server.yaml` — CARD data paths populated.
- `tests/test_forward_pass.py` — full pipeline integration test: ESM2Wrapper →
  SoftPromptModule → ClassifierHead → AMRLoss, 8M model, batch=2, parametrized
  over both injection modes. Confirms forward pass produces all four loss keys,
  and backward pass sends gradients to soft prompt + classifier while every
  ESM-2 parameter stays at `None`/zero grad — the frozen-backbone guarantee
  verified at the full-pipeline level, not just within `ESM2Wrapper` alone.
  Seeded before any module construction so weight init is reproducible too.
- `tests/` — 99/99 passing (32 data pipeline incl. split + 18 dataset + 32
  esm2_wrapper + 6 soft_prompt + 9 classifier + 2 end-to-end forward/backward).
- `docs/sessions/` — logs for all sessions to date, including this one.
- `docs/reviews/` — `/code-review` findings logged per session, with
  fixed/deferred status tracked so open items aren't lost.

## What's In Progress

Nothing actively mid-implementation. `split_dataset` complete; next is filling
in `configs/base.yaml`'s model/training sections (needs a few hyperparameter
decisions first — see Open Questions), then `train.py`.

## What's Not Started

In intended build order:

1. **`configs/base.yaml`, `local.yaml`, `cpu_server.yaml`, `gpu_server.yaml`** —
   model architecture section (`esm2_variant`, `device`, `injection_mode`) and
   training section (`batch_size`, `learning_rate`, `epochs`, optimizer) — none
   of these exist yet in any config file. No config-loading utility exists in
   the codebase yet either (nothing calls `yaml.safe_load` anywhere) — will
   need one before `train.py`/`run_training.py` can read `--config`.
2. **`src/training/train.py`** — main training loop with wandb logging; blocked
   on the config work above plus the open hyperparameter decisions
3. **`src/eval/evaluate.py`** — per-class F1, confusion matrix, CARD holdout eval
4. **`scripts/preprocess_card.py`** — one-time preprocessing entry point
5. **`scripts/run_training.py`** — training entry point (`--config` argument)
6. **`src/data/prodigal_runner.py`** — nucleotide → AA translation (needed for
   real-genome inference, not required for CARD FASTA training)

## Open Questions / Blockers

- **Numeric hyperparameters not yet decided:** learning rate, batch size,
  epochs, optimizer (Adam vs. AdamW + weight decay). CLAUDE.md's example config
  uses `lr=1e-4, batch_size=32, epochs=50` as a possible starting point, but
  nothing's been confirmed.
- **Ablation config structure not yet decided:** one `injection_mode` field
  hand-edited between the two ablation runs, vs. two dedicated config files
  (e.g. `gpu_server_internal.yaml` / `gpu_server_external.yaml`) so each run is
  independently reproducible from a single `--config` invocation per CLAUDE.md's
  "every training run must be fully specified by a config file" rule.
- **`configs/base.yaml`'s `loss:` section is not yet wired to anything.**
  `AMRLoss`'s constructor defaults (1.0/1.0/1.0) currently match it by
  coincidence. When `train.py` is written, it must construct
  `AMRLoss(**config["loss"])` explicitly (see the TODO comment in
  `src/training/loss.py`) rather than relying on the Python defaults — otherwise
  a future edit to the config's loss weights would silently have no effect on
  the actual run.
- **`docs/reviews/2026-07-05-classifier-loss-review.md`** has the full list of
  deferred code-review findings (test-coverage gaps, duplicated `input_dim`
  formula, doc nitpicks) — worth a pass before/around the first real GPU
  training run, none of them urgent before that.

## Recent Changes

1. **`split_dataset` implemented and tested**, then moved to `src/data/dataset.py`
   (`tests/test_dataset.py`) — train/val/test split on ARO accession, stratified
   by resistance_mechanism, 80/10/10 default, deterministic local RNG. 10 new
   unit tests plus 2 new integration tests against the real CARD dataset (full
   coverage, zero overlap, ratio bounds). This was identified as a genuine
   blocker for `train.py`: the split strategy was decided on 2026-07-02 but
   never implemented until now. Initially written in `card_parser.py`, then
   moved to `dataset.py` to keep `card_parser.py` scoped to parsing raw CARD
   files into `CARDRecord` objects only — split logic operates on
   already-parsed records and belongs with the rest of dataset preparation.
2. **`/code-review` run on the classifier+loss session** (8 finder angles, 9
   findings survived verification) — logged to
   `docs/reviews/2026-07-05-classifier-loss-review.md`. Two findings fixed
   immediately (BCE/CE reduction mismatch in `AMRLoss`, RNG seed order in
   `test_forward_pass.py`); the loss-weights-config gap partially addressed
   (config section added, wiring deferred to `train.py` with a TODO); the rest
   logged as deferred with reasoning for why each is safe to defer.
3. **`AMRLoss` implemented and tested** (`src/training/loss.py`) — combines
   BCEWithLogitsLoss (drug_class) + 2× CrossEntropyLoss (mechanism, family) into
   a weighted sum, with weights as constructor args (default 1.0 each). Handles
   the `drug_class_labels` → `drug_class` key mapping internally rather than
   renaming keys in `AMRDataset` or `ClassifierHead`.
4. **`tests/test_forward_pass.py` implemented** — first true end-to-end test of
   the full V1 model stack, parametrized over both injection modes, verifying
   both forward and backward passes.
5. **`ClassifierHead` implemented and tested** (prior session, 2026-07-05
   earlier) — shared trunk + three task heads, dict-shaped output.
6. **Loss weighting and train/val/test split resolved for V1** (prior session,
   2026-07-02) — equal loss weights (α = β = γ = 1); split on ARO accessions (not
   sequences), stratified by resistance mechanism, 80/10/10 ratio, fixed random
   seed. (Split design was resolved in this earlier session; the implementation
   itself landed in this session, item 1 above.)
