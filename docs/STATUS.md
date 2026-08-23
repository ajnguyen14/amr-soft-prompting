# AMR Soft Prompting — Project Status
_Last updated: 2026-08-23 10:25_

## Current Version

**V2 — Single-Head Retargeting + TA-Proximity Conditioning: training complete
per the finalized poster.** `docs/poster/poster.pdf` (finalized 2026-08-12)
reports finished results for all 6 single-head ablations (drug_class /
resistance_mechanism / amr_gene_family, each internal + external). No code
commits since `4ddb15d` (2026-08-08) — the codebase has been frozen while
the poster was assembled, and the project is now moving into handoff, not
active development. V1 unchanged from prior status (functionally complete).
V3 not started.

**Important gap for the handoff (see Open Questions below): this repo copy's
local `outputs/` and `wandb/` only contain artifacts for 2 of the 6 V2 runs**
(Run 1 external, Run 2 external). The other 4 runs' numbers appear in the
poster but have no local checkpoint or wandb cache on this machine
(`spark-833c`) — they ran on one of the other two GPU servers (`sjsu` or the
third RTX 3090 box, per CLAUDE.md's 3-server setup), and **that server has
since gone offline**, so those checkpoints/logs are currently unreachable —
not just uncopied. Confirmed by Aidan (2026-08-23).

## Completion

**V1: 100%, unchanged** (both ablation checkpoints trained and evaluated —
`i7o4eg5n` internal / `2rr2h1f9` external, 0.9087 / 0.9054 val gene-family
accuracy). **V2 code/config: 100%.** **V2 training: 6/6 runs complete
per the poster's reported numbers**, but **locally archived: 2/6** (see gap
above) — this repo directory does not currently hold reproducible artifacts
(checkpoint + eval) for Run 1 internal, Run 2 internal, or either Run 3
ablation.

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

1. **Recovering the 4 missing V2 run artifacts** (Run 1 internal, Run 2
   internal, Run 3 internal, Run 3 external) — blocked on the GPU server
   they ran on being offline; wandb.ai cloud is the only other lead. See
   Open Questions.
2. `src/data/prodigal_runner.py` — still an empty stub, deferred (unchanged,
   kept intentionally per CLAUDE.md's documented project structure).
3. V3 (KEGG/KofamKOALA, RAG) — not started, not scoped in detail.

## Open Questions / Blockers

- **Local artifact gap, currently unrecoverable (confirmed by Aidan,
  2026-08-23):** the poster reports finished numbers for all 6 V2
  ablations, but this repo directory only has checkpoints/wandb cache for
  2 of them (both external-injection runs). The other 4 ran on a different
  GPU server, and **that server has since gone offline** — so
  `outputs/task1_drugclass_internal/`, `outputs/task2_mechanism_internal/`,
  and both `outputs/task3_genefamily_{internal,external}/` cannot currently
  be pulled from it. The one remaining path to recover them is wandb.ai
  cloud (project `amr-soft-prompting`) — check whether those 4 runs synced
  there before the server went down; if not, the poster's numbers for
  those 4 tasks currently have no retrievable underlying checkpoint. Worth
  flagging to Andreopoulos/whoever inherits this if reproducing those
  specific results matters going forward — this isn't blocking the Drive
  handoff itself, just something the recipient should know is missing.
- GPU allocation / server state for the other two machines wasn't checked
  this session (no jobs were launched) — not urgent since nothing is
  currently queued.
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
5. **Discovered the local-artifact gap** (see Open Questions) while
   reconciling this repo's `outputs/`/`wandb/` against the poster's
   reported numbers.
