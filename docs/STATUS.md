# AMR Soft Prompting — Project Status
_Last updated: 2026-08-07 15:47_

## Current Version

**V2 — Single-Head Retargeting + TA-Proximity Conditioning.** Run 3's
sparse-signal blocker is now resolved: Aidan decided (rather than waiting
for a synchronous reply from Andreopoulos, given the poster deadline) to
collapse TA-proximity to the coarse 3-way categorical (`distance` /
`no_ta_locus` / `unknown`), option (1) from the three previously on the
table. All three V2 runs are now code-complete and config-complete.

- **Runs 1 & 2 (drug_class / resistance_mechanism, both conditioned on
  amr_gene_family): code-complete. Run 1 external is still training**
  (on `spark-833c`, started 02:32, still active as of this update); the
  other 3 Run 1/2 jobs not yet launched.
- **Run 3 (TA-proximity → amr_gene_family): unblocked and code-complete.**
  Step 4 no longer needs the (indefinitely blocked) distance-histogram bin
  edges — it's now a direct passthrough of Step 3's existing 3-way category.
  `configs/gpu_task3_genefamily_{internal,external}.yaml` written and
  load-verified; the full `train_v2` loop (build_v2_models → run_v2_epoch →
  checkpoint) has been exercised end-to-end on the 8M smoke model. **Not
  yet launched on GPU** — see What's Not Started.

## Completion

**V1: unchanged, functionally complete** (both ablation checkpoints trained
and evaluated). **V2 single-head restructuring (Runs 1/2): ~92%** — all
code, tests, and configs are done; Run 1 external is launched and training,
the other 3 jobs still need to be started. **V2 Run 3 (TA-proximity
conditioning): ~95%** — data layer, categorical join, configs, and training
wiring all done and tested; only the actual GPU launch (both ablations)
remains, same as the other 3 not-yet-launched Run 1/2 jobs.

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
    on this mapping. Now includes `'ta_proximity'` (Run 3's conditioning
    field, `'ce'`, batch key `'ta_proximity'`).
- **`build_v2_models`/`run_v2_epoch`/`train_v2` (`src/training/train.py`)**
  wired into `train.py main()`, dispatching on the presence of a `task`
  config section (else falls back to the untouched V1 `train()`). Confirmed
  generic enough that Run 3 required **zero changes** to this module — only
  the data layer (below) needed new code.
- **V2 eval path (`src/eval/evaluate.py`)**: `collect_predictions_v2`,
  `evaluate_single_label_target`, `evaluate_multi_label_target`, same
  `main()` dispatch pattern.
- **6 V2 configs, all load-verified**, parity rule followed (internal/
  external identical except `injection_mode`/`output_dir`/`wandb_run_name`),
  `batch_size: 24` (confirmed RTX 3090 ceiling, with a comment to bump to 32
  on `spark-833c`):
  - `gpu_task1_drugclass_{internal,external}.yaml`,
    `gpu_task2_mechanism_{internal,external}.yaml` (unchanged).
  - **New this session:** `gpu_task3_genefamily_{internal,external}.yaml`
    (`conditioning_field: ta_proximity`, `target_field: amr_gene_family`),
    with `paths.ta_proximity_results` pointing at
    `data/processed/ta_proximity_results.json`.
- **Run 3's TA-proximity data layer, newly wired this session:**
  - `CARDRecord.ta_proximity_category` (`src/data/card_parser.py`) — new
    field, default `""`. Populated only when `load_card_dataset` is called
    with the new `ta_proximity_path` argument, which joins
    `ta_proximity_results.json` onto records by ARO accession (missing
    accessions default to `'unknown'`).
  - `get_label_vocabularies` emits a `'ta_proximity'` vocab (exactly
    `['distance', 'no_ta_locus', 'unknown']`, sorted) only when at least one
    loaded record has a non-empty category — Run 1/2 callers (no
    `ta_proximity_path`) never see this key, so `AMRDataset` correctly omits
    the `'ta_proximity'` batch key for them too.
  - `AMRDataset` (`src/data/dataset.py`) conditionally builds
    `_ta_proximity_to_idx` and emits a `'ta_proximity'` long-scalar tensor
    key exactly when that vocab is present.
  - `scripts/preprocess_card.py` and `src/training/train.py`'s
    `build_dataloaders` both thread `paths.ta_proximity_results` through to
    `load_card_dataset` automatically — no separate Run-3-only code path.
  - **Validated against the real, full CARD + TA-proximity data on
    `spark-833c`**: loading all 6052 records with `ta_proximity_path` set
    reproduces the exact category counts from the fixed-code pipeline rerun
    (19 `distance`, 1933 `no_ta_locus`, 4100 `unknown`) — confirms the join
    logic and the reported numbers agree.
- **Full smoke suite: 253/253 passing** (was 241; +12 new tests for the
  TA-proximity join, `AMRDataset`'s `ta_proximity` key, `build_v2_models`
  with `conditioning_field='ta_proximity'`, and a full `train_v2` integration
  run for the Run 3 shape). No regressions to the V1 or Run 1/2 paths.
- **All 9 TA-proximity pipeline code-review findings fixed (commit
  `9bba25d`, prior session):**
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
- **TA-proximity pipeline (Steps 1 and 3) rerun end-to-end on `spark-833c`
  with the fixed code** (prior session), producing corrected, trustworthy
  numbers that this session's real-data validation (above) reproduced
  exactly:
  - Query universe (Step 1): 6052 ARO accessions queryable (had a CARD
    protein sequence to BLAST) across 676 representative groups; 352
    excluded up front as a **data-quality gap**, not folded into `unknown`.
  - BLAST coverage (Step 1): 1952/6052 accessions got a genomic-coordinate
    hit (32.3%).
  - Categorization (Step 3 / now also Run 3's Step 4 vocabulary directly),
    all as a fraction of the 6052-accession queryable universe:
    - `distance` (real same-replicon bp value, category kept, bp value
      dropped): **19 (0.31%)**.
    - `no_ta_locus` (mapped successfully, no TA locus on that replicon):
      1933 (31.9%).
    - `unknown` (genuine BLAST failure): 4100 (67.7%).
  - Distance histogram (bp), n=19: min=523, max=854495, p10=1717, p25=15818,
    p50=92656, p75=432500, p90=765686. No longer load-bearing for Step 4
    (fine-grained bins were dropped), kept here for reference only.

## What's In Progress

- **Run 1 external training** (`configs/gpu_task1_drugclass_external.yaml`)
  is running on `spark-833c` (started 02:32, still active as of this
  update).

## What's Not Started

1. **Launching all 4 remaining V2 GPU jobs**: Run 1 internal, both Run 2
   jobs, and both Run 3 jobs (`gpu_task3_genefamily_{internal,external}.yaml`)
   — 1 of 6 total V2 single-head ablations now running, 5 remain. Run 3's
   configs are new this session and have not yet been launched or had a
   preprocessed split artifact generated (training will fall back to
   parsing CARD directly on first launch, same as Run 1/2 did).
2. `src/data/prodigal_runner.py` — still an empty stub, deferred (unchanged).

## Open Questions / Blockers

- **Former TOP BLOCKER resolved:** Run 3's TA-proximity signal was too
  sparse (19/6052, 0.31%) for fine-grained distance bins. Resolved this
  session by collapsing to the coarse 3-way categorical
  (`distance`/`no_ta_locus`/`unknown`) — option (1) of the three previously
  on the table (see CLAUDE.md's TA-Proximity Pipeline section for the
  updated spec). Options (2) (expand TADB scope) and (3) (drop TA-proximity
  entirely) were not pursued. Aidan made this call directly rather than
  waiting on a synchronous reply from Andreopoulos, given the poster
  deadline — worth flagging to him after the fact since it's a real scope
  decision, not just an implementation detail.
- **GPU allocation for the remaining 5 V2 jobs not yet decided.** Need to
  check free GPU state on `spark-833c` (currently running Run 1 external
  plus several `ollama` processes consuming GPU memory), `sjsu`, and the
  third RTX 3090 box before launching more.
- Organism-dedup substitution rate (83%) and TADB Type I/III-VIII expansion
  are unchanged/still open, not repeated in detail here. One lower-severity
  (non-critical) code-review finding remains open: no BLAST-DB
  skip-if-exists (DBs are rebuilt from scratch on every rerun) — everything
  else from that review is fixed (see What's Working). Neither affects
  Run 3's now-collapsed categorical, since it no longer depends on precise
  bp distances.

## Recent Changes

1. **Resolved Run 3's sparse-signal blocker by collapsing TA-proximity to a
   coarse 3-way categorical** (`distance`/`no_ta_locus`/`unknown`), per
   Aidan's decision this session. Updated CLAUDE.md's TA-Proximity Pipeline
   Step 4 to document the collapse and drop the now-moot distance-bin plan.
2. **Wired Run 3's conditioning field end-to-end**: `CARDRecord
   .ta_proximity_category` + `load_card_dataset(..., ta_proximity_path=...)`
   (`src/data/card_parser.py`), a `'ta_proximity'` entry in
   `TARGET_FIELD_SPECS` and conditional `AMRDataset` batch key
   (`src/data/dataset.py`), and threading `paths.ta_proximity_results`
   through `scripts/preprocess_card.py` and `train.py`'s
   `build_dataloaders`. `build_v2_models`/`run_v2_epoch`/`train_v2` needed
   no changes — confirms they were built generically enough for all 3 runs,
   as originally intended.
3. **Wrote and load-verified `configs/gpu_task3_genefamily_
   {internal,external}.yaml`** — same parity rule as the other 4 V2 configs.
4. **Added 12 smoke tests** covering the TA-proximity join, `AMRDataset`'s
   conditional `ta_proximity` key, `build_v2_models` with
   `conditioning_field='ta_proximity'`, and a full `train_v2` run for the
   Run 3 shape (253/253 total passing, up from 241).
5. **Validated the join against the real, full CARD + TA-proximity data**
   on `spark-833c` — reproduces the exact 19/1933/4100 category counts from
   the earlier pipeline rerun.
6. Previous session: fixed all 9 TA-proximity pipeline code-review findings,
   reran Steps 1 and 3 with the fixed code, and launched Run 1 external
   (`gpu_task1_drugclass_external.yaml`) on `spark-833c`.
