# AMR Soft Prompting — Project Status
_Last updated: 2026-06-30 12:30_

## Current Version

**V1 — Core Pipeline**, mid-implementation. The data ingestion layer is complete and tested.
`AMRDataset` (PyTorch Dataset wrapping CARD records) is implemented but not yet committed.
ESM-2 wrapper, soft prompt module, classifier head, and training loop have not been started.

## Completion

**~25% of V1.** Breakdown:
- Data layer: ~90% (card_parser done; dataset.py done but uncommitted; prodigal_runner.py not started)
- Model layer: 0% (esm2_wrapper, soft_prompt, classifier all empty stubs)
- Training layer: 0% (train.py, loss.py empty stubs)
- Eval layer: 0% (evaluate.py empty stub)
- Scripts: 0% (preprocess_card.py, run_training.py empty stubs)
- Tests: ~35% of planned coverage written (data pipeline + dataset; soft prompt + forward pass not started)

## What's Working

- `src/data/card_parser.py` — parses CARD FASTA + `aro_index.tsv` into `CARDRecord` objects;
  builds sorted label vocabularies for drug class, resistance mechanism, and AMR gene family.
  Handles multi-drug-class semicolon splitting. Tested against real CARD v4.0.1 data (6052 records).
- `src/data/dataset.py` (`AMRDataset`) — wraps `CARDRecord` list into a PyTorch `Dataset`;
  encodes drug classes as multi-hot float32 tensors, resistance mechanism and AMR gene family
  as scalar long integer indices. Compatible with default `DataLoader` collation. *(Uncommitted)*
- `tests/test_data_pipeline.py` — 22/22 passing; covers header parsing, minimal fixtures,
  label vocabulary construction, and full CARD dataset (guarded when data absent).
- `tests/test_dataset.py` — written, covering tensor shapes, multi-hot encoding correctness,
  single-index label correctness, and DataLoader collation. *(Uncommitted)*
- `configs/gpu_server.yaml` — CARD data paths populated for the server.
- `conftest.py` + `pyproject.toml` — pytest configured; `src.*` imports resolve correctly.
- `CLAUDE.md` — comprehensive operating instructions, versioned roadmap, server escalation rules.

## What's In Progress

- Nothing actively mid-implementation at this moment. `AMRDataset` and its tests are written
  but need to be committed.

## What's Not Started

In intended build order:

1. **`src/data/prodigal_runner.py`** — wrap Prodigal CLI for nucleotide → amino acid translation
   (needed for real-genome inference, not required for CARD FASTA training)
2. **`src/models/esm2_wrapper.py`** — load ESM-2 frozen, tokenize sequences, extract
   mean-pooled embeddings
3. **`src/models/soft_prompt.py`** — the core contribution: encode CARD metadata vectors and
   condition ESM-2 embeddings via learnable soft prompt
4. **`src/models/classifier.py`** — MLP head consuming combined embedding + soft prompt
5. **`src/training/loss.py`** — BCEWithLogitsLoss (drug class) + CrossEntropyLoss (mechanism,
   gene family) combined
6. **`src/training/train.py`** — main training loop with wandb logging
7. **`src/eval/evaluate.py`** — per-class F1, confusion matrix, CARD holdout evaluation
8. **`scripts/preprocess_card.py`** — one-time preprocessing entry point
9. **`scripts/run_training.py`** — training entry point (`--config` argument)
10. **`tests/test_soft_prompt.py`** — soft prompt module shape smoke tests
11. **`tests/test_forward_pass.py`** — end-to-end forward pass (8M model, batch=2, CPU)
12. **`configs/base.yaml`, `local.yaml`, `cpu_server.yaml`** — populate with full config structure

## Open Questions / Blockers

- **Soft prompt design:** how exactly to encode mechanism + drug class vectors and inject them
  into ESM-2's representation space. This is the core novel contribution and needs an explicit
  design decision before `soft_prompt.py` is implemented. Options: prefix token injection,
  additive bias, or learned projection layer.
- **Multi-task loss weighting:** how to balance BCEWithLogitsLoss (drug class, multi-label)
  vs. CrossEntropyLoss (mechanism, gene family, single-label) in the combined training objective.
- **Prodigal integration priority:** for V1 evaluation we use the CARD FASTA directly (already
  amino acid), so `prodigal_runner.py` may not be needed until inference on raw nucleotide input.
- **Train/val/test split strategy:** CARD ARO accessions should be the splitting unit (not
  sequences) to prevent data leakage from multi-sequence ARO entries.
- **`configs/base.yaml` content:** needs to be populated before the training loop can be built.

## Recent Changes

1. **`AMRDataset` implemented** (`src/data/dataset.py`) — multi-hot drug class encoding for
   BCEWithLogitsLoss; integer indices for mechanism and gene family; sequences deferred as
   raw strings for ESM-2 tokenizer. Analysis of CARD v4.0.1 showed 57% of records have >1
   drug class, confirming multi-label as the correct default.
2. **`tests/test_dataset.py` written** — 20+ tests covering tensor shapes, dtype, multi-hot
   correctness, out-of-vocab handling, and single-index label alignment.
3. **`CLAUDE.md` updated** — added "Project Status File" section with template and instructions
   for end-of-session `docs/STATUS.md` updates.
4. **`card_parser.py` complete** (previous session) — `CARDRecord`, `load_card_dataset`,
   `get_label_vocabularies`; 22/22 tests passing.
5. **Project scaffold and CLAUDE.md established** (previous session) — three-environment model,
   versioned roadmap, server escalation rules, config system all documented.
