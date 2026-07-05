# AMR Soft Prompting — Project Status
_Last updated: 2026-07-05 00:00_

## Current Version

**V1 — Core Pipeline**, mid-implementation. Data layer complete. Model layer
complete. Combined loss and full-pipeline integration test implemented this
session. Next: `train.py` (blocked on `configs/base.yaml` model/training
sections).

## Completion

**~70% of V1.** Breakdown:
- Data layer: 100% (card_parser, AMRDataset, both tested)
- Model layer: 100% (esm2_wrapper, soft_prompt, classifier all done and tested)
- Training layer: ~50% (loss.py done and integration-tested; train.py empty stub)
- Eval layer: 0% (evaluate.py empty stub)
- Scripts: 0% (preprocess_card.py, run_training.py empty stubs)
- Tests: ~85% of planned coverage (data + dataset + esm2_wrapper + soft_prompt +
  classifier + end-to-end forward/backward pass, both injection modes)

## What's Working

- `src/data/card_parser.py` — CARD FASTA + ARO index → `CARDRecord` objects;
  label vocabularies for drug class, resistance mechanism, AMR gene family.
  Tested against real CARD v4.0.1 (6052 records).
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
- `src/training/loss.py` (`AMRLoss`) — BCEWithLogitsLoss (drug_class) +
  CrossEntropyLoss (resistance_mechanism, amr_gene_family), weighted sum with
  constructor-arg weights (default 1.0 each, matching the locked-in α=β=γ=1
  decision). Resolves the `drug_class_labels` → `drug_class` key-name mismatch
  between `AMRDataset` and `ClassifierHead` internally, without touching either.
  Returns a dict of per-task losses plus `total` for logging.
- `configs/base.yaml` — has a `classifier` section (`hidden_dim: 512`,
  `dropout: 0.1`). Model architecture and training sections still empty.
- `configs/gpu_server.yaml` — CARD data paths populated.
- `tests/test_forward_pass.py` — full pipeline integration test: ESM2Wrapper →
  SoftPromptModule → ClassifierHead → AMRLoss, 8M model, batch=2, parametrized
  over both injection modes. Confirms forward pass produces all four loss keys,
  and backward pass sends gradients to soft prompt + classifier while every
  ESM-2 parameter stays at `None`/zero grad — the frozen-backbone guarantee
  verified at the full-pipeline level, not just within `ESM2Wrapper` alone.
- `tests/` — 89/89 passing (22 data pipeline + 20 dataset + 32 esm2_wrapper +
  6 soft_prompt + 9 classifier + 2 end-to-end forward/backward).
- `docs/sessions/` — logs for all sessions to date, including this one.

## What's In Progress

Nothing actively mid-implementation. `loss.py` and `test_forward_pass.py`
complete; `train.py` is next.

## What's Not Started

In intended build order:

1. **`src/training/train.py`** — main training loop with wandb logging; blocked
   on `configs/base.yaml` getting model architecture and training hyperparameter
   sections (currently only `classifier:` exists)
2. **`src/eval/evaluate.py`** — per-class F1, confusion matrix, CARD holdout eval
3. **`scripts/preprocess_card.py`** — one-time preprocessing entry point
4. **`scripts/run_training.py`** — training entry point (`--config` argument)
5. **`src/data/prodigal_runner.py`** — nucleotide → AA translation (needed for
   real-genome inference, not required for CARD FASTA training)
6. **`configs/base.yaml`, `local.yaml`, `cpu_server.yaml`** — model architecture
   and training hyperparameter sections (only `classifier` section exists so far)

## Open Questions / Blockers

- **`configs/base.yaml`** still needs model architecture and training sections
  before `train.py` can be built (only `classifier` hyperparameters have been
  added so far). This is a hyperparameter decision, not pure implementation, so
  it's being treated as a separate step rather than bundled into a module task.

## Recent Changes

1. **`AMRLoss` implemented and tested** (`src/training/loss.py`) — combines
   BCEWithLogitsLoss (drug_class) + 2× CrossEntropyLoss (mechanism, family) into
   a weighted sum, with weights as constructor args (default 1.0 each). Handles
   the `drug_class_labels` → `drug_class` key mapping internally rather than
   renaming keys in `AMRDataset` or `ClassifierHead`.
2. **`tests/test_forward_pass.py` implemented** — first true end-to-end test of
   the full V1 model stack, parametrized over both injection modes. Extended
   beyond the CLAUDE.md forward-only checklist to also verify the backward pass:
   gradients reach soft prompt + classifier params, and confirm ESM-2 stays
   frozen at the integration level (mirroring `test_esm2_wrapper.py`'s
   single-module gradient test, now across the whole pipeline). 89/89 full suite
   passing.
3. **Caught a real key-naming mismatch before implementing** — the original spec
   for `AMRLoss` referenced `batch["resistance_mechanism_label"]` /
   `batch["amr_gene_family_label"]`, but `AMRDataset` actually returns those
   keys without the `_label` suffix. Corrected before writing the code rather
   than shipping a loss function that would silently `KeyError` against the
   real dataset.
4. **`ClassifierHead` implemented and tested** (prior session, 2026-07-05
   earlier) — shared trunk + three task heads, dict-shaped output.
5. **Loss weighting and train/val/test split resolved for V1** (prior session,
   2026-07-02) — equal loss weights (α = β = γ = 1); split on ARO accessions (not
   sequences), stratified by resistance mechanism, 80/10/10 ratio, fixed random
   seed.
