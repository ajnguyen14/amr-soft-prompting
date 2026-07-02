# AMR Soft Prompting — Project Status
_Last updated: 2026-07-02 15:00_

## Current Version

**V1 — Core Pipeline**, mid-implementation. Data layer complete. ESM-2 backbone
complete. Soft prompt design locked in following the 2026-07-02 meeting with
Andreopolous — unblocked, ready to implement `soft_prompt.py` next session.

## Completion

**~40% of V1.** Breakdown:
- Data layer: 100% (card_parser, AMRDataset, both tested)
- Model layer: ~30% (esm2_wrapper done; soft_prompt, classifier not started)
- Training layer: 0% (loss.py, train.py empty stubs)
- Eval layer: 0% (evaluate.py empty stub)
- Scripts: 0% (preprocess_card.py, run_training.py empty stubs)
- Tests: ~55% of planned coverage (data + dataset + esm2_wrapper; soft_prompt +
  forward pass not yet written)

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
- `tests/` — 72/72 passing (22 data pipeline + 20 dataset + 30 esm2_wrapper).
- `configs/gpu_server.yaml` — CARD data paths populated.
- `docs/sessions/` — retroactive session logs for all three sessions to date.

## What's In Progress

Nothing actively mid-implementation. Soft prompt design finalized 2026-07-02;
implementation starts next session.

## What's Not Started

In intended build order:

1. **`src/models/soft_prompt.py`** — design locked in 2026-07-02, ready to implement:
   - Injection modes: both internal and external run as an ablation (internal =
     hypothesis, external = baseline)
   - Mechanism encoding: `nn.Embedding` lookup, single integer → 1280-dim vector
   - Drug class encoding: `nn.Embedding` lookup with sum pooling over active class
     embeddings → 1280-dim vector
   - 2 soft prompt tokens; output dim = `embed_dim` = 1280, no projection layer needed
2. **`tests/test_soft_prompt.py`** — soft prompt module shape smoke tests
3. **`src/models/classifier.py`** — MLP head; input width depends on injection_mode
   and soft prompt output dim
4. **`src/training/loss.py`** — BCEWithLogitsLoss (drug class) + CrossEntropyLoss
   (mechanism, family) combined
5. **`src/training/train.py`** — main training loop with wandb logging
6. **`tests/test_forward_pass.py`** — end-to-end forward pass (8M model, batch=2, CPU)
7. **`src/eval/evaluate.py`** — per-class F1, confusion matrix, CARD holdout eval
8. **`scripts/preprocess_card.py`** — one-time preprocessing entry point
9. **`scripts/run_training.py`** — training entry point (`--config` argument)
10. **`src/data/prodigal_runner.py`** — nucleotide → AA translation (needed for
    real-genome inference, not required for CARD FASTA training)
11. **`configs/base.yaml`, `local.yaml`, `cpu_server.yaml`** — full config content

## Open Questions / Blockers

- **`configs/base.yaml`** needs content before the training loop can be built.

## Recent Changes

1. **Andreopolous meeting (2026-07-02)** — soft prompt design approved; Aidan
   proceeding with implementation. Supervisor emphasis: dataset clarity and
   reproducibility, since he intends to continue this work with other students
   toward publication.
2. **Soft prompt design locked in** — both injection modes run as an ablation
   (internal = hypothesis, external = baseline); `nn.Embedding` lookups for
   mechanism (single int → 1280-dim) and drug class (sum-pooled over active
   classes → 1280-dim); 2 soft prompt tokens; output dim = `embed_dim` = 1280,
   no projection layer needed.
3. **Loss weighting resolved** — equal weights (α = β = γ = 1) for V1;
   `BCEWithLogitsLoss` (drug class), `CrossEntropyLoss` (mechanism, family);
   tuning deferred to V2.
4. **Train/val/test split strategy resolved** — split on ARO accessions (not
   sequences) to prevent data leakage; stratified by resistance mechanism;
   80/10/10 ratio; fixed random seed for reproducibility.
