# AMR Soft Prompting — Project Status
_Last updated: 2026-08-24 19:34_

## Current Version

**V2 — Single-Head Retargeting + TA-Proximity Conditioning: training complete
per the finalized poster.** `docs/poster/poster.pdf` (finalized 2026-08-12)
reports finished results for all 6 single-head ablations (drug_class /
resistance_mechanism / amr_gene_family, each internal + external). No code
commits since `4ddb15d` (2026-08-08) — the codebase has been frozen while
the poster was assembled, and the project is now moving into handoff, not
active development. V1 unchanged from prior status (functionally complete).
V3 not started.

**Handoff artifact gap is now resolved.** The other 4 V2 runs (Run 1
internal, Run 2 internal, Run 3 internal, Run 3 external) trained on
`sjsu-1`, which had gone offline (see prior note) — it came back online
2026-08-24 and Aidan pulled all 4 missing `outputs/` checkpoint folders
directly from it to his laptop's copy of this repo. **The laptop copy used
for the Google Drive handoff now has all 6/6 V2 run checkpoints.** Note this
server's (`spark-833c`) own local `outputs/`/`wandb/` was *not* updated by
that pull (Aidan pulled `sjsu-1` → laptop directly) — see Completion below
for the scope distinction.

## Completion

**V1: 100%, functionally complete**, but **the `i7o4eg5n`-equivalent
checkpoint on this server specifically is currently corrupted** — see the
incident note in Open Questions. Not a project-completion issue, just a
`spark-833c`-local file state issue; the original checkpoint survives in
Aidan's laptop copy (pulled before the incident) and possibly wandb.ai.
**V2 code/config: 100%.** **V2 training: 6/6 runs complete** per the
poster's reported numbers, and **6/6 now have recovered checkpoints on
Aidan's laptop** (the handoff copy) as of 2026-08-24. **This server's
(`spark-833c`) own local `outputs/`/`wandb/` still only has 2/6** (Run 1
external, Run 2 external) — the other 4 were pulled `sjsu-1` → laptop
directly, bypassing this machine, so this repo directory alone is not the
complete artifact set. That's expected and fine — the laptop copy is the
one being shipped.

## What's Working

- Everything from V1 (unchanged): `card_parser.py`, `dataset.py`,
  `esm2_wrapper.py`, `SoftPromptModule`, `ClassifierHead`, `AMRLoss`,
  `preprocess_card.py`, `run_training.py`, `load_config`.
- Full V2 single-head architecture (`SingleFieldSoftPrompt`,
  `SingleTargetClassifierHead`, `SingleTargetLoss`,
  `compute_single_target_metrics`, `TARGET_FIELD_SPECS`, `build_v2_models`/
  `run_v2_epoch`/`train_v2`, V2 eval path) — unchanged since last session,
  confirmed generic enough to have supported all 3 tasks without further
  changes.
- All 6 V2 configs (`gpu_task{1,2,3}_{genefamily,drugclass,mechanism}_
  {internal,external}.yaml`), parity rule followed.
- TA-proximity data layer (`CARDRecord.ta_proximity_category`, the
  `'ta_proximity'` `TARGET_FIELD_SPECS` entry, conditional `AMRDataset`
  batch key) — validated against the real, full CARD + TA-proximity data.
- **Full smoke suite: 253/253 passing**, reconfirmed this session on
  `spark-833c`. No regressions.
- **Locally-verified V2 results (this machine's `outputs/`/`wandb/`):**
  - Run 1 external (`v2-task1-drugclass-external`, `wandb` run
    `p9f8pdmn`, 50 epochs): drug_class subset accuracy 92.3%, micro-F1
    94.2%. Checkpoint: `outputs/task1_drugclass_external/best_model.pt`.
  - Run 2 external (`v2-task2-mechanism-external`, `wandb` run
    `oe13u15r`, 50 epochs): resistance_mechanism accuracy 99.5%.
    Checkpoint: `outputs/task2_mechanism_external/best_model.pt`.
- **Poster-reported V2 results (not locally archived — see gap above):**
  - Drug class: internal 85.4% / external 92.3% (external matches local).
  - Mechanism: internal 99.5% / external 99.5% (external matches local).
  - Gene family (← TA-proximity): internal 90.9% / external 86.5%, vs.
    an 18.8% naive-lookup baseline (the ~72pp gap the poster's conclusion
    is built on).
  - TA-proximity conditioning composition: 67.7% unknown, 31.9%
    no_ta_locus, 0.3% distance (n=19) — matches the pipeline numbers
    already recorded here from the prior session.
- **Poster finalized**: `docs/poster/poster.pdf` + `poster.md` + 9 figures
  (2026-08-12). Full results writeup, all 3 tasks × both injection modes,
  label-leakage confound explainer, TA-proximity coverage-gap discussion.
  Untracked in git — repo handoff is happening as a raw folder copy via
  Google Drive, not a git clone, so this was left uncommitted by choice.
- `notebooks/exploration/poster_figures.ipynb` — generated the poster
  figures. Also untracked, same reasoning.

## What's In Progress

Nothing actively running. No commits or wandb activity since 2026-08-08;
the project has been dormant since the poster was finalized on 2026-08-12
while moving toward handoff.

## What's Not Started

1. `src/data/prodigal_runner.py` — still an empty stub, deferred (unchanged,
   kept intentionally per CLAUDE.md's documented project structure).
2. V3 (KEGG/KofamKOALA, RAG) — not started, not scoped in detail.
3. (Optional, low priority) Mirroring the 4 recovered `outputs/` folders
   from Aidan's laptop back onto `spark-833c` itself, if a fully consistent
   6/6 copy on this server ever matters — not needed for the Drive handoff,
   which ships from the laptop copy.

## Open Questions / Blockers

- **Incident (2026-08-23/24): cleanup accidentally re-triggered a stale
  automation, overwriting `spark-833c`'s local V1 internal checkpoint.**
  `scripts/check_memory_headroom.sh` runs via cron every 10 minutes and
  gates a one-time auto-launch of the old V1 internal run
  (`configs/gpu_server_internal.yaml`) behind a fire-once guard file
  (`outputs/.internal_launched`). During Drive-handoff cleanup on
  2026-08-23, that guard file (and `outputs/.memory_watch_state`) were
  deleted as presumed-inert transient state — **this reset the guard**, and
  the next cron tick relaunched the V1 internal run, which ran undetected
  for over a day (started 10:40, discovered ~19:20 the next day) and
  **overwrote `outputs/internal/best_model.pt`** (the `i7o4eg5n`-equivalent
  V1 checkpoint) at least once. The rogue run's wandb logging also never
  produced a `config.yaml`/summary locally despite the long runtime — its
  health is questionable independent of the overwrite issue. **Resolution:**
  both the wrapper process and its training subprocess were killed
  (confirmed dead, GPU utilization back to 0%); the fire-once guard files
  were already re-written by the script at launch time, so the cron job
  will *not* auto-relaunch again. **Net effect:** this server's copy of the
  V1 internal checkpoint is currently in a bad/partial state, but the
  Drive-handoff laptop copy is unaffected — it was archived at 10:41 on
  2026-08-23, one minute *before* the rogue run started, so it still has
  the original good checkpoint. No action needed for the handoff; only
  matters if someone does further work on `spark-833c` itself and expects
  `outputs/internal/best_model.pt` to be the original V1 result.
  **Lesson for future cleanup passes on this repo:** dotfile/state markers
  living next to a cron-driven script (`outputs/.internal_launched`,
  `.memory_watch_state`) are live automation state, not disposable — check
  `crontab -l` for anything that references a file before deleting it.
- Organism-dedup substitution rate (83%) and TADB Type I/III-VIII expansion
  remain open, unchanged from before. One lower-severity code-review finding
  remains open: no BLAST-DB skip-if-exists.
- Organism-dedup substitution rate (83%) and TADB Type I/III-VIII expansion
  remain open, unchanged from before. One lower-severity code-review finding
  remains open: no BLAST-DB skip-if-exists.

## Recent Changes

1. **Poster finalized** (`docs/poster/poster.pdf`, 2026-08-12) — full
   results writeup for all 3 V2 tasks across both injection modes, plus
   the label-leakage confound and TA-proximity coverage-gap explainers.
2. **No code changes since `4ddb15d`** (2026-08-08, "collapse Run 3
   TA-proximity to 3-way categorical and wire end-to-end") — codebase has
   been frozen since then; this session's changes are handoff prep only.
3. **Repo cleanup for Google Drive handoff** (this session): removed
   `__pycache__`/`.pytest_cache`, a duplicate copy of the poster figures
   under `outputs/poster_figures/` (byte-identical to
   `docs/poster/figures/`), and transient `wait_and_run.sh`/memory-watch
   state files (`outputs/.internal_launched`, `.memory_watch_state`,
   `memory_watch.log`, `internal_run.out`). Kept `data/raw/` (BLAST DB +
   RefSeq, 788M), `wandb/` local cache, and all `outputs/` checkpoints —
   Aidan wants these included in the Drive upload as-is.
4. **`docs/poster/` and `notebooks/` intentionally left untracked in git**
   — the handoff is a raw folder copy via Google Drive, not a git clone,
   so committing them wasn't necessary. Same reasoning applied to the
   uncommitted `requirements.txt` addition (`jupyter`, `ipykernel`,
   `nbconvert`, `nbformat` — needed for the poster-figures notebook).
5. **Discovered, then closed, the local-artifact gap**: reconciling this
   repo's `outputs/`/`wandb/` against the poster's reported numbers turned
   up 4 of 6 V2 runs missing locally (trained on `sjsu-1`, which was
   offline at the time). `sjsu-1` came back online 2026-08-24, and Aidan
   pulled all 4 missing `outputs/` checkpoint folders directly to his
   laptop's copy of this repo — the Drive-handoff copy now has all 6/6 V2
   run checkpoints. This server's own `outputs/`/`wandb/` was not updated
   by that pull (see Completion).
6. **Incident: cleanup-triggered auto-relaunch of the V1 internal run**
   overwrote `spark-833c`'s local `outputs/internal/best_model.pt` — caused
   by deleting a cron job's fire-once guard file during cleanup, believing
   it to be inert. Process killed once discovered; cron will not
   auto-relaunch again (guard file was already restored by the script at
   launch time). Does not affect the Drive-handoff laptop copy, which was
   archived before the overwrite happened. Full writeup in Open Questions.
