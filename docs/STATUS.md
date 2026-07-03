# AMR Soft Prompting — Project Status
_Last updated: 2026-07-02 17:30_

## Current Version

**V1 — Core Pipeline**, mid-implementation. Data layer complete. ESM-2 backbone
complete. Soft prompt module implemented and tested against the settled design
from the 2026-07-02 Andreopolous meeting. Next: `classifier.py`.

## Completion

**~50% of V1.** Breakdown:
- Data layer: 100% (card_parser, AMRDataset, both tested)
- Model layer: ~65% (esm2_wrapper + soft_prompt done; classifier not started)
- Training layer: 0% (loss.py, train.py empty stubs)
- Eval layer: 0% (evaluate.py empty stub)
- Scripts: 0% (preprocess_card.py, run_training.py empty stubs)
- Tests: ~65% of planned coverage (data + dataset + esm2_wrapper + soft_prompt;
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
- `src/models/soft_prompt.py` (`SoftPromptModule`) — encodes resistance mechanism
  (`nn.Embedding` lookup) and drug class (multi-hot vector times the embedding weight
  matrix, mathematically equivalent to sum-pooling over active class embeddings) into
  2 soft prompt tokens of shape `(B, 2, embed_dim)`. All parameters trainable.
- `tests/` — 78/78 passing (22 data pipeline + 20 dataset + 30 esm2_wrapper +
  6 soft_prompt).
- `configs/gpu_server.yaml` — CARD data paths populated.
- `docs/sessions/` — retroactive session logs for all three sessions to date, plus
  the 2026-07-02 design + implementation session.

## What's In Progress

Nothing actively mid-implementation. `soft_prompt.py` complete; `classifier.py` is
next.

## What's Not Started

In intended build order:

1. **`src/models/classifier.py`** — MLP head; input width depends on injection_mode
   and soft prompt output dim
2. **`src/training/loss.py`** — BCEWithLogitsLoss (drug class) + CrossEntropyLoss
   (mechanism, family) combined
3. **`src/training/train.py`** — main training loop with wandb logging
4. **`tests/test_forward_pass.py`** — end-to-end forward pass (8M model, batch=2, CPU)
5. **`src/eval/evaluate.py`** — per-class F1, confusion matrix, CARD holdout eval
6. **`scripts/preprocess_card.py`** — one-time preprocessing entry point
7. **`scripts/run_training.py`** — training entry point (`--config` argument)
8. **`src/data/prodigal_runner.py`** — nucleotide → AA translation (needed for
   real-genome inference, not required for CARD FASTA training)
9. **`configs/base.yaml`, `local.yaml`, `cpu_server.yaml`** — full config content

## Open Questions / Blockers

- **`configs/base.yaml`** needs content before the training loop can be built.

## Recent Changes

1. **`SoftPromptModule` implemented and tested** (`src/models/soft_prompt.py`,
   `tests/test_soft_prompt.py`) — mechanism via `nn.Embedding` lookup, drug class
   via multi-hot vector times the embedding weight matrix, 2 output tokens at
   `embed_dim`. 6/6 new tests passing; full suite 78/78.
2. **Caught and fixed a real integration bug during implementation** —
   `AMRDataset` produces drug class as a multi-hot `(B, num_drug_classes)` float32
   vector, not a list of active indices. A literal `nn.Embedding` lookup on that
   tensor would have been wrong. Fixed with a matmul against the embedding weight
   matrix (mathematically equivalent to sum-pooling over active classes) — no
   changes needed to `AMRDataset`.
3. **Andreopolous meeting (2026-07-02)** — soft prompt design approved; Aidan
   proceeding with implementation. Supervisor emphasis: dataset clarity and
   reproducibility, since he intends to continue this work with other students
   toward publication.
4. **Loss weighting and train/val/test split resolved for V1** — equal loss
   weights (α = β = γ = 1); split on ARO accessions (not sequences), stratified
   by resistance mechanism, 80/10/10 ratio, fixed random seed.
