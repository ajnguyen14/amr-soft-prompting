# AMR Soft Prompting — Project Status
_Last updated: 2026-07-17 (spark-833c)_

## Current Version

**V1 — Core Pipeline**, code complete and just re-fixed for label leakage.
Both ablation checkpoints from the prior session are now stale and need
retraining before the poster comparison table can be trusted.

## Completion

**V1 code is ~100% complete**, but the trained artifacts are not: the
label-leakage fix below changes `ClassifierHead`'s parameter shapes, so both
`outputs/{internal,external}/best_model.pt` checkpoints (50/50 epochs, scored
in the last session) no longer load against the current model code and need
to be retrained from scratch. Retraining has **not** been started yet (held
off deliberately, pending a decision on when to spend the multi-day GPU time).

## What's Working

Everything listed in the prior status entry still holds structurally
(`card_parser.py`, `dataset.py` split/`AMRDataset`, `esm2_wrapper.py`,
`soft_prompt.py`, `preprocess_card.py`, `run_training.py`, `load_config`, all
five configs), plus this session's fix:

- **Label-leakage fix landed.** `resistance_mechanism` and `drug_class` are
  no longer classifier prediction targets — `SoftPromptModule` still
  conditions on their ground-truth values (unchanged, still the intended
  "known metadata as context" design), but `ClassifierHead` now has only an
  `amr_gene_family_head`, `AMRLoss` only scores `amr_gene_family`, and
  `compute_metrics`/`evaluate.py` only report `amr_gene_family_accuracy`
  (+ macro-F1, top-confused-pairs, confusion-matrix CSV in `evaluate.py`).
  This makes V1's remaining question ("does internal or external injection
  work better") measurable on a task that isn't circular: the model has to
  actually read the ESM-2 sequence representation to predict gene family,
  since gene family is never fed into the soft prompt.
- Files touched: `src/models/classifier.py`, `src/training/loss.py`,
  `src/eval/metrics.py`, `src/training/train.py` (`build_models`/`run_epoch`),
  `src/eval/evaluate.py` (deleted `evaluate_resistance_mechanism`,
  `evaluate_drug_class`, `plot_confusion_matrix`,
  `plot_multilabel_confusion_grid` — all dead code once those two heads were
  removed), `configs/base.yaml` (`loss:` section now just
  `weight_amr_gene_family`). `soft_prompt.py` itself is untouched.
- All 5 smoke test modules touching this surface
  (`test_classifier.py`, `test_metrics.py`, `test_train.py`,
  `test_evaluate.py`, `test_forward_pass.py`, `test_config.py`) updated to
  match. Full suite: **133/133 passing** on spark-833c (CPU-side smoke tests,
  8M model).

## What's In Progress

- Nothing actively running. Next action is a decision on when to spend the
  GPU time to retrain both ablations against the fixed classifier.

## What's Not Started

1. **Retraining both ablation configs** (`configs/gpu_server_internal.yaml`,
   `configs/gpu_server_external.yaml`) against the fixed `ClassifierHead`.
   Both prior checkpoints are now unusable. Historically ~2.2 days
   (external) / ~4.6 days (internal) wall-clock on spark-833c.
2. **`src/data/prodigal_runner.py`** — nucleotide → AA translation. Still an
   empty stub, explicitly deferred to V2 by Aidan (2026-07-09): V1
   training/eval only ever consumes CARD's pre-translated amino acid FASTA,
   so this isn't needed for V1 functional completeness.
3. Poster write-up — now blocked on the retrain above, since the only
   ablation numbers that exist (from the pre-fix runs) measured a leaked
   signal and shouldn't be used.

## Open Questions / Blockers

- **Label leakage — fixed in code, not yet re-measured.** Previously:
  `resistance_mechanism`/`drug_class` were both fed into `SoftPromptModule`
  as conditioning input *and* scored as classifier prediction targets, so
  those two metrics measured how well the classifier decoded its own
  soft-prompt embedding rather than anything learned from the sequence.
  Fixed this session by dropping both as prediction targets — `amr_gene_family`
  is now the only classifier output, matching the one task that was never fed
  into the soft prompt. This was a scoped decision (see conversation
  2026-07-17): keep `soft_prompt.py`'s conditioning inputs unchanged (no PI
  sign-off needed, since the "novel contribution" module itself wasn't
  touched), and only trim the classifier/loss/metrics/eval surface. The
  previous internal-vs-external comparison table (resistance_mechanism
  accuracy, amr_gene_family accuracy/macro-F1, drug_class F1) is retired —
  only amr_gene_family accuracy/macro-F1 will be reported going forward, and
  that requires both ablations to be retrained.
- **V2 scope update (2026-07-17, Aidan's decision, not yet in CLAUDE.md):**
  TA loci (TADB 3.0) is being pulled forward from V3 into V2, alongside
  RefSeq — V2 will map CARD metadata + TA loci onto RefSeq. CLAUDE.md's
  Versioned Roadmap section still lists TA loci as V3-only; Aidan is updating
  it himself. Relevant to a future soft-prompt redesign: TA loci + RefSeq
  signal is a candidate for soft-prompt conditioning that doesn't overlap
  with CARD prediction targets, avoiding this same class of leakage
  structurally rather than by dropping tasks.
- **Is internal mode's underperformance real or confounded by
  gradient-checkpointing?** Unresolved from before, and now moot until
  retrained on the fixed classifier — the prior gap was measured partly on
  leaked metrics.
- **`spark-833c` is a shared, unified-memory box** — GPU memory pool is the
  same pool as system RAM (`nvidia-smi` reports device-level `Memory-Usage`
  as "Not Supported" for this reason). Relevant whenever retraining starts.
- `docs/reviews/2026-07-05-classifier-loss-review.md` — remaining deferred
  items are low-severity doc/style nitpicks; unchanged.

## Recent Changes

1. **Label-leakage fix**: removed `resistance_mechanism`/`drug_class` as
   `ClassifierHead` prediction targets and from `AMRLoss`/`compute_metrics`/
   `evaluate.py`; `SoftPromptModule` unchanged. `configs/base.yaml`'s `loss:`
   section trimmed to match.
2. Updated 6 test files (`test_classifier.py`, `test_metrics.py`,
   `test_train.py`, `test_evaluate.py`, `test_forward_pass.py`,
   `test_config.py`) to match the new single-task classifier. Full suite
   passing (133/133).
3. Both existing ablation checkpoints (`outputs/{internal,external}/best_model.pt`)
   are now stale and will need retraining before any new results are
   reported.
4. Noted a V2 scope change (TA loci moved from V3 into V2, paired with
   RefSeq) that Aidan is reflecting in CLAUDE.md separately.
5. Retraining deliberately not started yet — held off pending a separate
   go-ahead given the multi-day GPU cost.
