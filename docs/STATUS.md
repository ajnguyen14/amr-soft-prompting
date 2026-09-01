# AMR Soft Prompting — Project Status
_Last updated: 2026-09-01 16:30_

## Current Version

**V2 — Single-Head Retargeting + TA-Proximity Conditioning, extended with negative-control ablations across all 3 tasks. Core V2 + controls are training-complete; handoff artifacts are current.**

Note on this entry: two independent STATUS.md updates were written on 2026-08-24, from two different machines (`spark-833c` and `sjsu`), each unaware of the other — this reconciles both into one timeline, then brings it current to today.

- By **2026-08-12**, `docs/poster/poster.pdf` was finalized, reporting results for all 6 core single-head ablations (drug_class / resistance_mechanism / amr_gene_family, each internal + external). The codebase was considered frozen at `4ddb15d`, and the project was moving into a Google Drive handoff (raw folder copy, not a git clone).
- A **wandb reconciliation pass** (`wandb.Api()`, 2026-08-19) confirmed all 8 core runs (2 V1 + 6 V2) had actually finished, correcting an earlier (2026-08-07) claim in this file that only 1/6 V2 jobs were launched — that entry was never re-synced after later launches. Two stale pre-leakage-fix wandb runs (`snm1i141`, `zz67qllg`) were also found in the project and excluded from all reporting.
- During Drive-handoff cleanup on `spark-833c` (**2026-08-23**), 4 of 6 V2 runs (trained on `sjsu-1`, offline at the time) were found missing from that machine's local `outputs/`/`wandb/`. `sjsu-1` came back online 2026-08-24 and all 4 were pulled directly to Aidan's laptop, completing the Drive-handoff copy (6/6). The same cleanup pass also caused an incident that corrupted `spark-833c`'s *local* copy of the V1 internal checkpoint — see Open Questions. The Drive-handoff laptop copy is unaffected.
- **Then, per Andreopoulos's direction (2026-08-24), work resumed past the "frozen" state**: negative-control ablations (`NullSoftPrompt`, a fixed non-learned zero token replacing real conditioning) were added to test whether `amr_gene_family`/`ta_proximity` conditioning contributes any signal beyond the bare architecture. Originally scoped to Run 3 only, this was extended to all three tasks (see CLAUDE.md's "Negative Control Runs" section). Built and launched on `sjsu`: Task 3's pair first (2026-08-24), then Task 1's pair (2026-08-25), then Task 2's pair (2026-08-25, interrupted; successfully retried 2026-08-26).
- **As of today (2026-09-01)**: all 6 negative-control runs are finished with results (see Completion). The code (`NullSoftPrompt`, `build_v2_models`/`run_v2_epoch` wiring, 6 configs, 7 new tests) was committed on `sjsu` (`24e49dd`) and reconciled with origin via rebase. `outputs/`, `logs/`, and `wandb/` for all runs (core + controls) were verified fully synced to Aidan's laptop via `rsync` dry-run (no diffs beyond NTFS permission-bit noise).

## Completion

**V1: 100%, functionally complete.** `spark-833c`'s local copy of the `i7o4eg5n`-equivalent checkpoint was corrupted by the cron incident (see Open Questions) — not a project-completion issue; the original survives in the Drive-handoff laptop copy and wandb.

**V2 core (Runs 1-3, both injection modes): 100% of training**, poster-reported and wandb-confirmed:
- Run 1 (drug_class ← gene_family): internal (`mlvam3jw`) 0.854 subset acc / 0.919 micro-F1; external (`p9f8pdmn`) 0.923 / 0.942.
- Run 2 (resistance_mechanism ← gene_family): internal (`xpyy4zde`) 0.9952; external (`oe13u15r`) 0.9952. **Unresolved leakage-shape flag** — see Open Questions.
- Run 3 (ta_proximity → gene_family): internal (`x7f7gvh3`) 0.9087; external (`p4kl9z5r`) 0.8654.
- Gap: only V1 has a held-out `evaluate.py` pass with a full confusion matrix; none of the 6 core V2 runs have been run through `evaluate.py` — numbers above are training-loop final-epoch val metrics only.

**V2 negative controls: 6/6 runs complete** (task1/2/3 × internal/external, `conditioning_field: "none"`), finished 2026-08-26, results verified from `logs/` this session:
- Task 1 (drug_class): internal 0.8349 / external 0.8349 subset acc — **identical between injection modes**, worth a closer look (plausible under an all-zero token regardless of injection site, but not yet confirmed as expected vs. a wiring bug).
- Task 2 (resistance_mechanism): internal 0.9952 / external 0.9952 (from the successful 2026-08-26 retries — first 2026-08-25 attempts were killed mid-run, stale wandb dirs `404g947p`/`i5q0e1h7` still present, not real results).
- Task 3 (gene_family): internal 0.9071 / external 0.9006.
- **Not yet compared against their conditioned counterparts** in a single table — that comparison (control vs. Run 1/2/3) is the actual point of these runs and hasn't been written up yet. See What's Not Started.

## What's Working

- Everything from V1 (unchanged): `card_parser.py`, `dataset.py`, `esm2_wrapper.py`, `SoftPromptModule`, `ClassifierHead`, `AMRLoss`, `preprocess_card.py`, `run_training.py`, `load_config`, both retrained V1 ablation checkpoints (`i7o4eg5n` internal / `2rr2h1f9` external, 0.9087 / 0.9054 val gene-family accuracy).
- **V2 single-head architecture, built generically for Runs 1-3.** New classes added *alongside* (not replacing) their V1 counterparts, specifically so the two existing V1 checkpoints' state_dicts stay loadable:
  - `SingleFieldSoftPrompt` (`src/models/soft_prompt.py`) — 1-token categorical embedding for any single conditioning field.
  - `NullSoftPrompt` (`src/models/soft_prompt.py`, added 2026-08-24) — fixed all-zero token as a registered buffer (not `nn.Parameter`), used for the negative-control runs. Confirmed no trainable parameters.
  - `SingleTargetClassifierHead` (`src/models/classifier.py`) — shared-trunk MLP with a configurable `target_name`/`num_classes` head.
  - `SingleTargetLoss` (`src/training/loss.py`) — dispatches to `BCEWithLogitsLoss` ('bce', Run 1's multi-label drug_class) or `CrossEntropyLoss` ('ce', Run 2/3's single-label targets).
  - `compute_single_target_metrics` (`src/eval/metrics.py`) — argmax accuracy for 'ce' targets; 0.5-thresholded subset accuracy + micro-F1 for 'bce' targets.
  - `TARGET_FIELD_SPECS` (`src/data/dataset.py`) — single source of truth mapping each label field to its `AMRDataset` batch key and loss type. Includes `'ta_proximity'` (Run 3's conditioning field).
- **`build_v2_models`/`run_v2_epoch`/`train_v2` (`src/training/train.py`)**, dispatching on the presence of a `task` config section. `conditioning_field == "none"` wires a `NullSoftPrompt` instead of looking `'none'` up in `TARGET_FIELD_SPECS`/label vocabularies.
- **V2 eval path (`src/eval/evaluate.py`)**: `collect_predictions_v2`, `evaluate_single_label_target`, `evaluate_multi_label_target`.
- **12 V2 configs, all load-verified**, parity rule followed (internal/external identical except `injection_mode`/`output_dir`/`wandb_run_name`), `batch_size: 24` (confirmed RTX 3090 ceiling):
  - `gpu_task{1,2,3}_{drugclass,mechanism,genefamily}_{internal,external}.yaml` (core, 6 files).
  - `gpu_task{1,2,3}_{drugclass,mechanism,genefamily}_noconditioning_{internal,external}.yaml` (negative controls, 6 files, `conditioning_field: "none"`).
- **Run 3's TA-proximity data layer**: `CARDRecord.ta_proximity_category` (`src/data/card_parser.py`), joined via `load_card_dataset(..., ta_proximity_path=...)`; `get_label_vocabularies` emits the 3-way `'ta_proximity'` vocab only when populated, so Run 1/2 callers are unaffected; `AMRDataset` conditionally emits the batch key. Validated against the full 6052-record CARD + TA-proximity data on `spark-833c`, reproducing the exact category counts from the pipeline rerun (19 `distance`, 1933 `no_ta_locus`, 4100 `unknown`).
- **Full smoke suite: 39/39 passing for this session's touched modules** (`test_soft_prompt.py`, `test_train.py`), reconfirmed on `sjsu` before committing `24e49dd`; prior full-suite count was 260/260 (253 + 7 new for `NullSoftPrompt`/`conditioning_field="none"`). No regressions to V1, Run 1/2, or Run 3 paths.
- **All 9 TA-proximity pipeline code-review findings fixed** (commit `9bba25d`): query-universe/BLAST-failure disambiguation, atomic RefSeq genome writes, `used_own_accession` bias flag threaded through to results, deterministic BLAST tie-break, duplicate-taxonomy-id hard failure, superseded-script docstring, per-group error handling + checkpointing, config-driven e-value threshold.
- **TA-proximity pipeline (Steps 1 & 3) rerun end-to-end with the fixed code**: 6052 queryable accessions (352 excluded up front, no CARD sequence), 1952/6052 BLAST coverage (32.3%), category split `distance` 19 (0.31%) / `no_ta_locus` 1933 (31.9%) / `unknown` 4100 (67.7%).
- **Poster finalized** (`docs/poster/poster.pdf` + `poster.md` + 9 figures, 2026-08-12): full results writeup, label-leakage confound explainer, TA-proximity coverage-gap discussion. Untracked in git by choice (Drive handoff is a raw folder copy).
- **`notebooks/exploration/results_presentation.ipynb`** — Colab notebook pulling run results live via `wandb.Api()`, built for a 2026-08-20 presentation to Bill. Untracked in git.
- **Handoff artifacts fully synced to Aidan's laptop** (`C:\Users\aidan\Documents\GitHub\amr-soft-prompting`, via WSL2): `outputs/`, `logs/`, `wandb/` verified complete via `rsync -avzn --itemize-changes` (2026-09-01) — no missing or size-differing files, only NTFS permission-bit noise. Includes both the original 6-core-run gap-recovery (spark-833c → laptop, 2026-08-24) and today's full sync from `sjsu`.

## What's In Progress

Nothing actively running as of 2026-09-01. All negative-control runs finished 2026-08-26; code committed and being reconciled with origin (this rebase) today.

## What's Not Started

1. **Write up the control-vs-conditioned comparison** — the actual point of the negative-control runs (does `amr_gene_family`/`ta_proximity` conditioning beat a zero token?). Numbers exist in this file's Completion section but haven't been compared side-by-side or interpreted.
2. **Investigate Task 1 noconditioning's identical internal/external accuracy** (0.8349 both) — confirm this is an expected property of an all-zero token under both injection modes, not a wiring bug that makes `injection_mode` a no-op for the null case.
3. **Held-out `evaluate.py` pass for all 12 V2 runs** (6 core + 6 control) — only V1 has confusion-matrix-level eval artifacts.
4. **Leakage sanity-check for Run 2** — both injection modes landed at ~99.5% accuracy, matching the shape of the pre-fix leakage bug. Not yet checked whether `resistance_mechanism` is trivially predictable from `amr_gene_family` alone.
5. Clean up stale wandb run dirs from Task 2's killed first attempt (`run-20260825_105032-404g947p`, `run-20260825_105032-i5q0e1h7`) — harmless but risk being mistaken for real results later.
6. `src/data/prodigal_runner.py` — still an empty stub, deferred (unchanged).
7. Mirroring the 4 recovered `outputs/` folders from Aidan's laptop back onto `spark-833c` itself — optional, not needed for any current handoff.
8. V3 (KEGG/KofamKOALA, RAG) — not started, not scoped in detail.

## Open Questions / Blockers

- **Incident (2026-08-23/24, `spark-833c`): cleanup accidentally re-triggered a stale automation, overwriting that server's local V1 internal checkpoint.** `scripts/check_memory_headroom.sh` runs via cron every 10 minutes and gates a one-time auto-launch of the old V1 internal run behind a fire-once guard file (`outputs/.internal_launched`). During Drive-handoff cleanup, that guard file (and `outputs/.memory_watch_state`) were deleted as presumed-inert — this reset the guard, and the next cron tick relaunched the V1 internal run, which ran undetected for over a day and overwrote `outputs/internal/best_model.pt` on `spark-833c`. **Resolved**: both processes killed, guard files already re-written at launch time so cron will not auto-relaunch. The Drive-handoff laptop copy is unaffected (archived one minute before the rogue run started). **Lesson**: dotfile/state markers next to a cron-driven script are live automation state, not disposable — check `crontab -l` before deleting anything like this again.
- **Run 2's ~99.5% accuracy (both modes) needs a leakage sanity-check** before being reported as a clean result — see What's Not Started.
- **Task 1 noconditioning's identical 0.8349 internal/external result** needs the same kind of scrutiny — see What's Not Started.
- **TA-proximity's 3-way categorical collapse (2026-08-07) was a solo scope call**, made without a synchronous go-ahead from Andreopoulos given the poster deadline — still worth flagging to him after the fact if not already done.
- Organism-dedup substitution rate (83%) and TADB Type I/III-VIII expansion remain open. One lower-severity code-review finding remains open: no BLAST-DB skip-if-exists (DBs rebuilt from scratch on every rerun).
- This local `sjsu` branch was 2 commits behind `origin/v2-ta-proximity` when this session started (both docs-only STATUS.md commits from the `spark-833c` handoff session) — being reconciled via rebase right now; this file's content is the result of that reconciliation.

## Recent Changes

1. **Reconciled two independently-written 2026-08-24 STATUS.md histories** (this session, 2026-09-01) — `spark-833c`'s poster/handoff/incident narrative and `sjsu`'s wandb-reconciliation/negative-control narrative had diverged from a shared base (`4ddb15d`) without either side knowing about the other. Merged into one timeline (see Current Version) during a `git rebase` conflict resolution.
2. **Extended negative-control ablations from Run 3 only to all 3 tasks**: 6 total noconditioning runs (task1/2/3 × internal/external) built, launched on `sjsu`, and finished by 2026-08-26. Task 2's first attempt (2026-08-25) was interrupted mid-run and successfully retried.
3. **Committed the negative-control code** (`24e49dd`, this session): `NullSoftPrompt`, `build_v2_models`/`run_v2_epoch` wiring, 6 new configs, 7 new smoke tests (39/39 passing on touched modules), CLAUDE.md's "Negative Control Runs" section.
4. **Verified full handoff-artifact sync to Aidan's laptop** (this session): `outputs/`, `logs/`, `wandb/` confirmed complete via `rsync` dry-run over Tailscale (`sjsu` at `100.85.43.58`) — closes the gap between what's on `sjsu` and what's on the laptop for both core and control runs.
5. **Incident + recovery** (`spark-833c`, 2026-08-23/24): cleanup-triggered cron auto-relaunch overwrote the local V1 internal checkpoint on `spark-833c` (Drive-handoff laptop copy unaffected); separately, 4/6 V2 runs missing from `spark-833c`'s local artifacts were recovered by pulling directly from `sjsu-1` to the laptop once it came back online.
6. **Poster finalized** (`docs/poster/poster.pdf`, 2026-08-12) — full results writeup for all 3 V2 tasks across both injection modes, plus the label-leakage confound and TA-proximity coverage-gap explainers.
7. Previous session: resolved Run 3's sparse-signal blocker by collapsing TA-proximity to a coarse 3-way categorical; wired Run 3's conditioning field end-to-end; fixed all 9 TA-proximity pipeline code-review findings; reran Steps 1 and 3 with the fixed code.
