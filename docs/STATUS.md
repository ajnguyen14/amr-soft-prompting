# AMR Soft Prompting — Project Status
_Last updated: 2026-07-05 00:00_

## Current Version

**V1 — Core Pipeline**, mid-implementation. Data layer complete. ESM-2 backbone
complete. Soft prompt module complete. Classifier head implemented this session.
Next: `loss.py`.

## Completion

**~60% of V1.** Breakdown:
- Data layer: 100% (card_parser, AMRDataset, both tested)
- Model layer: 100% (esm2_wrapper, soft_prompt, classifier all done and tested)
- Training layer: 0% (loss.py, train.py empty stubs)
- Eval layer: 0% (evaluate.py empty stub)
- Scripts: 0% (preprocess_card.py, run_training.py empty stubs)
- Tests: ~75% of planned coverage (data + dataset + esm2_wrapper + soft_prompt +
  classifier; forward pass not yet written)

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
- `configs/base.yaml` — now has a `classifier` section (`hidden_dim: 512`,
  `dropout: 0.1`). Model architecture and training sections still empty.
- `configs/gpu_server.yaml` — CARD data paths populated.
- `tests/` — 87/87 passing (22 data pipeline + 20 dataset + 32 esm2_wrapper +
  6 soft_prompt + 9 classifier).
- `docs/sessions/` — logs for all sessions to date, including this one.

## What's In Progress

Nothing actively mid-implementation. `classifier.py` complete; `loss.py` is next.

## What's Not Started

In intended build order:

1. **`src/training/loss.py`** — BCEWithLogitsLoss (drug class) + CrossEntropyLoss
   (mechanism, family) combined, equal weights (α = β = γ = 1), consuming the
   dict shape `ClassifierHead.forward()` returns
2. **`src/training/train.py`** — main training loop with wandb logging
3. **`tests/test_forward_pass.py`** — end-to-end forward pass (8M model, batch=2, CPU)
4. **`src/eval/evaluate.py`** — per-class F1, confusion matrix, CARD holdout eval
5. **`scripts/preprocess_card.py`** — one-time preprocessing entry point
6. **`scripts/run_training.py`** — training entry point (`--config` argument)
7. **`src/data/prodigal_runner.py`** — nucleotide → AA translation (needed for
   real-genome inference, not required for CARD FASTA training)
8. **`configs/base.yaml`, `local.yaml`, `cpu_server.yaml`** — model architecture
   and training hyperparameter sections (only `classifier` section exists so far)

## Open Questions / Blockers

- **`configs/base.yaml`** still needs model architecture and training sections
  before the training loop can be built (only `classifier` hyperparameters have
  been added so far).

## Recent Changes

1. **`ClassifierHead` implemented and tested** (`src/models/classifier.py`,
   `tests/test_classifier.py`) — shared trunk + three task heads
   (drug_class/resistance_mechanism/amr_gene_family), dict-shaped output. 9/9 new
   tests passing; full suite 87/87.
2. **`configs/base.yaml` populated for the first time** — `classifier.hidden_dim:
   512`, `classifier.dropout: 0.1`. Scope kept narrow to classifier hyperparameters
   only; model/training sections remain open.
3. **Confirmed multi-task design before implementing** — reviewed
   `esm2_wrapper.py`'s mode-dependent output width and the 2026-07-02 loss
   weighting decision (α = β = γ = 1 across drug_class/mechanism/family) to settle
   on a shared-trunk, three-head architecture rather than a single output layer.
4. **`SoftPromptModule` implemented and tested** (prior session, 2026-07-02) —
   mechanism via `nn.Embedding` lookup, drug class via multi-hot vector times the
   embedding weight matrix, 2 output tokens at `embed_dim`.
5. **Loss weighting and train/val/test split resolved for V1** (prior session,
   2026-07-02) — equal loss weights (α = β = γ = 1); split on ARO accessions (not
   sequences), stratified by resistance mechanism, 80/10/10 ratio, fixed random
   seed.
