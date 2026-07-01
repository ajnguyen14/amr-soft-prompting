# AMR Soft Prompting — Project Status
_Last updated: 2026-06-30 13:00_

## Current Version

**V1 — Core Pipeline**, mid-implementation. Data layer complete. ESM-2 backbone
complete. Blocked on soft_prompt.py design conversation with Andreopolous before
the model layer can proceed.

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

Nothing actively mid-implementation. Blocked on soft_prompt.py design.

## What's Not Started

In intended build order:

1. **`src/models/soft_prompt.py`** ← **BLOCKED** on design conversation with Andreopolous
   - How to encode mechanism + drug class as continuous vectors
   - Whether to evaluate internal mode, external mode, or both as an ablation
   - Output dimensionality (must equal `embed_dim` = 1280 for 650M training model,
     or a projection layer is needed)
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

- **Soft prompt design (BLOCKING):** Three decisions needed from Andreopolous:
  1. Run both injection modes as an ablation, or commit to one?
  2. How to encode CARD metadata (mechanism, drug class) as continuous vectors?
     Options: learned lookup table, fixed one-hot projection, or something else.
  3. Does the soft prompt output dimension match `embed_dim` (simplest), or does
     it use a different size requiring a projection in `ESM2Wrapper`?
- **Train/val/test split:** should split on ARO accessions (not sequences) to prevent
  data leakage from multi-sequence entries.
- **Multi-task loss weighting:** how to balance BCE (drug class, multi-label) vs.
  CrossEntropy (mechanism, family) in the combined objective.
- **`configs/base.yaml`** needs content before the training loop can be built.

## Recent Changes

1. **`ESM2Wrapper` implemented** (`src/models/esm2_wrapper.py`) — both internal and
   external injection modes, verified frozen params, correct residue-only mean pooling.
   Key implementation finding: this version of HuggingFace transformers bypasses the
   embedding layer entirely when `inputs_embeds` is passed, requiring explicit
   pre-processing of word embeddings in internal mode.
2. **`tests/test_esm2_wrapper.py` written** — 32 tests; includes a gradient test
   confirming the frozen-param contract holds during a real backward pass.
3. **`AMRDataset` committed** (`src/data/dataset.py`) — multi-hot drug class encoding;
   57% of CARD records have >1 drug class so multi-label is the default, not the edge case.
4. **Session documentation established** — `docs/sessions/` with retroactive logs for
   all three sessions; `docs/STATUS.md` updated each session per CLAUDE.md instructions.
