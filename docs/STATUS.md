# AMR Soft Prompting — Project Status
_Last updated: 2026-08-07 10:26_

## Current Version

**V2 — Single-Head Retargeting + TA-Proximity Conditioning.** Given the
approaching poster deadline, work continues to prioritize Runs 1 & 2
(unblocked) over Run 3 (blocked on a data-sparsity decision — see
Blockers). Aidan plans to email Andreopoulos about the Run 3 blocker rather
than wait for a synchronous decision before the poster is due.

- **Runs 1 & 2 (drug_class / resistance_mechanism, both conditioned on
  amr_gene_family): code-complete. Run 1 external is now actually launched**
  (training on `spark-833c`, in progress as of this update); the other 3
  Run 1/2 jobs not yet launched.
- **Run 3 (TA-proximity conditioning): still blocked** on the sparse-signal
  finding (see Blockers). The 9 code-review findings against this pipeline
  are now all fixed, and the pipeline has been **rerun end-to-end with the
  fixed code** — see corrected numbers below. Bottom line: the sparsity
  blocker is confirmed, not an artifact of the bug — real-distance coverage
  is unchanged at 19 accessions (0.3%).

## Completion

**V1: unchanged, functionally complete** (both ablation checkpoints trained
and evaluated). **V2 single-head restructuring (Runs 1/2): ~92%** — all
code, tests, and configs are done; Run 1 external is launched and training,
the other 3 jobs still need to be started. **V2 TA-proximity data layer
(Run 3's input): ~78%** — all 9 code-review findings fixed and the pipeline
rerun end-to-end, producing trustworthy coverage numbers (see below); Step 4
(categorical distance-bin embedding) still blocked on the sparse-signal
finding.

## What's Working

- Everything from V1 (unchanged): `card_parser.py`, `dataset.py`,
  `esm2_wrapper.py`, `SoftPromptModule`, `ClassifierHead`, `AMRLoss`,
  `preprocess_card.py`, `run_training.py`, `load_config`, both retrained V1
  ablation checkpoints (`i7o4eg5n` internal / `2rr2h1f9` external, 0.9087 /
  0.9054 val gene-family accuracy).
- **V2 single-head architecture, built generically for Runs 1-3.** New
  classes added *alongside* (not replacing) their V1 counterparts,
  specifically so the two existing V1 checkpoints' state_dicts stay
  loadable:
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
    mapping each label field to its `AMRDataset` batch key and loss type,
    shared by train.py, evaluate.py, and metrics.py so the three can't drift
    on this mapping.
- **`build_v2_models`/`run_v2_epoch`/`train_v2` (`src/training/train.py`)**
  wired into `train.py main()`, dispatching on the presence of a `task`
  config section (else falls back to the untouched V1 `train()`).
- **V2 eval path (`src/eval/evaluate.py`)**: `collect_predictions_v2`,
  `evaluate_single_label_target`, `evaluate_multi_label_target`, same
  `main()` dispatch pattern.
- **4 Run 1/2 configs, all load-verified**, parity rule followed
  (internal/external identical except `injection_mode`), `batch_size: 24`
  (confirmed RTX 3090 ceiling, with a comment to bump to 32 on `spark-833c`).
  Run 3's config pair intentionally not written yet — still blocked.
- **Full smoke suite: 241/241 passing**, no regressions to the V1 path.
- **All 9 TA-proximity pipeline code-review findings fixed (commit
  `9bba25d`, this session):**
  - *Critical (result integrity):*
    1. Reported `unknown` rate no longer conflates "never BLASTed" (352
       accessions with no FASTA sequence) with "BLAST genuinely failed" —
       `run_blast_coordinate_mapping.py` now writes the exact query universe
       it attempted (`blast_query_universe.json`) and `run_ta_proximity.py`
       reads that instead of recomputing it independently, eliminating both
       the miscategorization and the duplicate recomputation between the two
       scripts.
    2. `refseq_fetch.py` genome writes are now validated (non-empty FASTA)
       and atomic (temp file + `os.replace`) — a truncated/error response
       can no longer be permanently cached as a valid fetch.
    3. `used_own_accession` (the substitution-genome bias flag, affecting
       83% of entries) now flows through `BlastHit` → `TAProximityResult`
       instead of being computed and discarded, so downstream results can be
       audited for this bias.
  - *Reproducibility:*
    4. BLAST best-hit tie-break is now deterministic (bitscore, then a
       stable coordinate key) instead of relying on subprocess output order.
    5. ARO accessions mapping to >1 taxonomy_id now raise loudly instead of
       silently picking one arbitrarily.
  - *Hygiene:*
    6. `card_tadb_matcher.py` docstrings now state it's superseded/not part
       of the live pipeline.
    7. Per-group error handling + incremental checkpointing to
       `blast_hits_output` — one bad representative genome no longer aborts
       the whole ~738-group run.
    8. E-value threshold is now config-driven (`configs/ta_proximity_refseq.yaml`)
       instead of a bare literal.
  - Full suite re-verified at 241/241 passing after the fixes, no
    regressions.
- **TA-proximity pipeline (Steps 1 and 3) rerun end-to-end on `spark-833c`
  with the fixed code**, producing corrected, trustworthy numbers:
  - Query universe (Step 1): 6052 ARO accessions queryable (had a CARD
    protein sequence to BLAST) across 676 representative groups; 352
    excluded up front as a **data-quality gap**, not folded into `unknown`
    anymore (this is the fix from finding #1).
  - BLAST coverage (Step 1): 1952/6052 accessions got a genomic-coordinate
    hit (32.3%).
  - Categorization (Step 3), all as a fraction of the 6052-accession
    queryable universe:
    - `distance` (real same-replicon bp value): **19 (0.31%)** — unchanged
      from the pre-fix count, confirming the sparsity blocker is real, not
      an artifact of the unknown-rate bug.
    - `no_ta_locus` (mapped successfully, no TA locus on that replicon):
      1933 (31.9%).
    - `unknown` (genuine BLAST failure): 4100 (67.7%) — this is the
      corrected, apples-to-apples figure. The old 69.5% figure was
      `(4100 genuine failures + 352 never-BLASTed) / 6404 total`; the fix
      separates the 352 out as a distinct data-quality-gap count instead of
      counting them as `unknown`.
  - Distance histogram (bp), n=19: min=523, max=854495, p10=1717, p25=15818,
    p50=92656, p75=432500, p90=765686. This is the real histogram Step 4's
    bin edges must eventually be derived from, if TA-proximity survives as
    Run 3's conditioning input (see Blockers).

## What's In Progress

- **Run 1 external training** (`configs/gpu_task1_drugclass_external.yaml`)
  is running on `spark-833c` (started 02:32, still active as of this
  update).
- Nothing in progress on the TA-proximity pipeline — the fixed-code rerun
  (below) completed this session.

## What's Not Started

1. **Launching Run 1 internal + both Run 2 GPU jobs** — 1 of 4 Run 1/2 jobs
   now running, 3 remain.
2. **Run 3's Step 4 (categorical distance-bin embedding)** — still blocked,
   unchanged (see Blockers).
3. `src/data/prodigal_runner.py` — still an empty stub, deferred (unchanged).

## Open Questions / Blockers

- **TOP BLOCKER, still open and now confirmed with corrected numbers: Run
  3's TA-proximity signal is too sparse to use as originally scoped.** Fixed
  rerun gives 19/6052 queryable accessions (0.31%) with a real same-replicon
  distance value — essentially unchanged from the pre-fix 19/6404 (0.3%)
  figure, so this is a real biological/data-coverage limit, not an artifact
  of the unknown-rate bug. Aidan is emailing Andreopoulos about this rather
  than waiting for a synchronous decision, given the poster deadline; Runs
  1/2 don't depend on this and were prioritized instead. Options on the
  table remain unchanged: (1) collapse to a coarse 3-way categorical
  (`distance`/`no_ta_locus`/`unknown`, effectively what Step 3 already
  outputs, skipping fine-grained bins entirely), (2) expand TADB scope
  beyond Type II, (3) drop TA-proximity as Run 3's conditioning input.
- **GPU allocation for the remaining 3 Run 1/2 jobs not yet decided.** Need
  to check free GPU state on `spark-833c` (currently running Run 1 external
  plus several `ollama` processes consuming GPU memory), `sjsu`, and the
  third RTX 3090 box before launching more.
- Organism-dedup substitution rate (83%) and TADB Type I/III-VIII expansion
  are unchanged/still open, not repeated in detail here. One lower-severity
  (non-critical) code-review finding remains open: no BLAST-DB
  skip-if-exists (DBs are rebuilt from scratch on every rerun, including the
  one in progress now) — everything else from that review is fixed (see
  What's Working).

## Recent Changes

1. **Fixed all 9 TA-proximity pipeline code-review findings** (commit
   `9bba25d`): 3 critical result-integrity bugs (unknown-rate conflation,
   unvalidated/non-atomic RefSeq writes, discarded substitution-bias flag),
   2 reproducibility bugs (nondeterministic tie-break, silent multi-taxonomy
   collision), 4 hygiene items (stale docstring, no per-group checkpointing,
   hardcoded e-value). Full suite 241/241 passing, no regressions.
2. **Reran Steps 1 and 3 of the TA-proximity pipeline** with the fixed code:
   32.3% BLAST coverage (1952/6052 queryable), corrected unknown rate 67.7%
   (was 69.5% pre-fix), and the sparsity blocker confirmed unchanged at
   19/6052 (0.31%) real distances. See What's Working for the full
   breakdown and the distance histogram.
3. **Launched Run 1 external** (`gpu_task1_drugclass_external.yaml`) on
   `spark-833c` — the first of the 4 Run 1/2 GPU jobs to actually run.
4. Previous session: built the V2 single-head architecture generically for
   Runs 1-3, wired `train_v2`/eval path into `train.py`/`evaluate.py`, wrote
   and load-verified the 4 Run 1/2 configs, added ~54 smoke tests (241/241
   total passing).
