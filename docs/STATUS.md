# AMR Soft Prompting — Project Status
_Last updated: 2026-08-07 01:44_

## Current Version

**V2 — Single-Head Retargeting + TA-Proximity Conditioning.** Given the
approaching poster deadline, work this session prioritized Runs 1 & 2
(unblocked) over Run 3 (blocked on a data-sparsity decision — see
Blockers). Aidan plans to email Andreopoulos about the Run 3 blocker rather
than wait for a synchronous decision before the poster is due.

- **Runs 1 & 2 (drug_class / resistance_mechanism, both conditioned on
  amr_gene_family): code-complete, not yet launched on a GPU server.**
- **Run 3 (TA-proximity conditioning): still blocked**, unchanged from last
  session's finding (see Blockers).

## Completion

**V1: unchanged, functionally complete** (both ablation checkpoints trained
and evaluated). **V2 single-head restructuring (Runs 1/2): ~90%** — all
code, tests, and configs are done; only actually launching and monitoring
the 4 GPU jobs remains. **V2 TA-proximity data layer (Run 3's input):
unchanged at ~70%**, Step 4 still blocked on the sparse-signal finding from
last session.

## What's Working

- Everything from V1 (unchanged): `card_parser.py`, `dataset.py`,
  `esm2_wrapper.py`, `SoftPromptModule`, `ClassifierHead`, `AMRLoss`,
  `preprocess_card.py`, `run_training.py`, `load_config`, both retrained V1
  ablation checkpoints (`i7o4eg5n` internal / `2rr2h1f9` external, 0.9087 /
  0.9054 val gene-family accuracy).
- **V2 single-head architecture, built generically for Runs 1-3 (this
  session).** New classes added *alongside* (not replacing) their V1
  counterparts, specifically so the two existing V1 checkpoints' state_dicts
  stay loadable:
  - `SingleFieldSoftPrompt` (`src/models/soft_prompt.py`) — 1-token
    categorical embedding for any single conditioning field.
  - `SingleTargetClassifierHead` (`src/models/classifier.py`) — shared-trunk
    MLP with a configurable `target_name`/`num_classes` head.
  - `SingleTargetLoss` (`src/training/loss.py`) — dispatches to
    `BCEWithLogitsLoss` ('bce', Run 1's multi-label drug_class) or
    `CrossEntropyLoss` ('ce', Run 2/3's single-label targets).
  - `compute_single_target_metrics` (`src/eval/metrics.py`) — argmax
    accuracy for 'ce' targets; 0.5-thresholded subset accuracy + micro-F1 for
    'bce' targets (plain accuracy isn't well-defined for multi-label).
  - `TARGET_FIELD_SPECS` (`src/data/dataset.py`) — single source of truth
    mapping each label field to its `AMRDataset` batch key and loss type
    (`drug_class` → `drug_class_labels`/`bce`; `resistance_mechanism` and
    `amr_gene_family` → themselves/`ce`), shared by train.py, evaluate.py,
    and metrics.py so the three can't drift on this mapping. Caught a real
    bug during testing: `drug_class`'s batch key differs from its field name,
    which would have `KeyError`'d if `SingleTargetLoss`/metrics read
    `batch[target_name]` directly instead of `batch[batch_key]`.
- **`build_v2_models`/`run_v2_epoch`/`train_v2` (`src/training/train.py`,
  this session).** Task-configurable training path, reading
  `config['task']['conditioning_field']`/`['target_field']`.
  `train.py main()` dispatches to `train_v2()` if the loaded config has a
  `task` section, else falls back to the untouched V1 `train()` — the
  `--config` file alone decides which pipeline runs.
- **V2 eval path (`src/eval/evaluate.py`, this session).**
  `collect_predictions_v2`, `evaluate_single_label_target` (macro-F1 +
  confusion matrix, generalizing the old `evaluate_amr_gene_family`), and
  `evaluate_multi_label_target` (subset accuracy + micro/macro-F1 + per-class
  F1 CSV, since a confusion matrix doesn't apply to multi-label). Same
  `main()` dispatch pattern as train.py.
- **4 new configs, all load-verified**: `configs/gpu_task1_drugclass_{internal,external}.yaml`
  (Run 1), `configs/gpu_task2_mechanism_{internal,external}.yaml` (Run 2).
  Internal/external pairs identical except `injection_mode`, per the parity
  rule. `batch_size: 24` (confirmed RTX 3090 ceiling), with a comment to bump
  to 32 if launched on `spark-833c` instead. Run 3's config pair intentionally
  not written yet — still blocked.
- **Full smoke suite: 241/241 passing**, including ~54 new tests across
  `test_soft_prompt.py`, `test_classifier.py`, the new `test_loss.py`,
  `test_metrics.py`, `test_train.py`, and `test_evaluate.py`. No regressions
  to the V1 path. `test_train.py`'s V2 integration test drives `main()`
  through its actual `--config` CLI entry point (writing a temp config dir
  with a sibling `base.yaml`), not just the internal functions directly.
- **`/code-review high` run against the TA-proximity pipeline this session**
  (separate from Runs 1/2, which don't touch this code at all). 10 findings;
  3 flagged critical for result integrity, not yet fixed:
  1. `scripts/run_ta_proximity.py:78` — the reported 69.5% `unknown` rate
     conflates "never BLASTed" (352 accessions with no FASTA sequence) with
     "BLAST genuinely failed," overstating the real failure rate.
  2. `src/data/refseq_fetch.py:107` — fetched genome FASTA is written and
     marked "succeeded" with no validation it's well-formed/non-empty, and
     the write isn't atomic; a truncated/error response is permanently
     cached as valid with no error ever surfaced.
  3. `scripts/run_blast_coordinate_mapping.py:67` — `refseq_representative.py`'s
     `used_own_accession` flag (whether an ARO was BLASTed against its own
     genome vs. a substituted strain's, affecting 83% of entries) is computed
     then discarded, so downstream TA-proximity results can't be audited for
     this bias.
  7 lower-severity findings (nondeterministic tie-break, missing cardinality
  guard, dead code, no BLAST-batch checkpointing, hardcoded e-value, no
  BLAST-db skip-if-exists, duplicate recomputation across scripts) also not
  yet fixed — full list available on request, not reproduced here.

## What's In Progress

- Nothing actively running. Next concrete step is launching the 4 Run 1/2
  GPU jobs (task1/task2 × internal/external) — ideally in parallel across
  the 3 GPU-capable servers, per CLAUDE.md's note that 4 total GPUs across
  3 machines make this feasible instead of queuing serially. Free-GPU state
  not yet checked this session.

## What's Not Started

1. **Launching/monitoring the 4 Run 1/2 GPU training jobs** — code and
   configs are ready; nothing has actually run on a GPU yet.
2. **Run 3's Step 4 (categorical distance-bin embedding)** — still blocked,
   unchanged from last session (see Blockers).
3. **The 3 critical TA-proximity pipeline integrity fixes** found by this
   session's code review (see What's Working) — not urgent for Runs 1/2,
   but should be resolved before citing Run 3/TA-proximity numbers in the
   poster or the planned Andreopoulos email.
4. `src/data/prodigal_runner.py` — still an empty stub, deferred (unchanged).

## Open Questions / Blockers

- **TOP BLOCKER, unchanged from last session: Run 3's TA-proximity signal is
  very likely too sparse to use as originally scoped.** Only 19/6404 (0.3%)
  ARO accessions get a real same-replicon distance value (and that number is
  itself somewhat inflated per code-review finding #1 above — see What's
  Working). Aidan is emailing Andreopoulos about this rather than waiting
  for a synchronous decision, given the poster deadline; Runs 1/2 don't
  depend on this and were prioritized instead. Options on the table remain
  unchanged: (1) collapse to a coarse 3-way categorical, (2) expand TADB
  scope beyond Type II, (3) drop TA-proximity as Run 3's conditioning input.
- **3 critical TA-proximity pipeline bugs found by code review, not yet
  fixed** — see What's Working for the list. Isolated from Runs 1/2 (fully
  separate code path), but affect the trustworthiness of any Run 3/sparsity
  numbers cited in the poster or the Andreopoulos email.
- **GPU allocation for the 4 new jobs not yet decided.** Need to check free
  GPU state on `spark-833c`, `sjsu`, and the third RTX 3090 box before
  launching, and watch for `ollama` memory contention on `spark-833c`
  (flagged last session).
- Organism-dedup substitution rate (83%), TADB Type I/III-VIII expansion, and
  the other previously-open items are unchanged from last session — not
  repeated here.

## Recent Changes

1. **Built the V2 single-head architecture generically for Runs 1-3**:
   `SingleFieldSoftPrompt`, `SingleTargetClassifierHead`, `SingleTargetLoss`,
   `compute_single_target_metrics`, and `TARGET_FIELD_SPECS` as the shared
   field→batch-key/loss-type mapping. Added alongside (not replacing) the V1
   classes to keep the two existing trained checkpoints loadable.
2. **Wired `build_v2_models`/`run_v2_epoch`/`train_v2` into `train.py`** and
   the parallel V2 path into `evaluate.py`, both dispatched from `main()` by
   the presence of a `task` config section.
3. **Wrote and load-verified 4 new configs** for Runs 1 and 2
   (internal/external pairs), following the parity rule.
4. **Added ~54 new smoke tests**; full suite 241/241 passing, no regressions.
5. **Ran `/code-review high` against the TA-proximity pipeline**; found 3
   critical result-integrity issues (unknown-rate conflation, unvalidated
   RefSeq fetch writes, discarded substitution-bias flag) not yet fixed.
