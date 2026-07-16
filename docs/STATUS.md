# AMR Soft Prompting — Project Status
_Last updated: 2026-07-15 21:40_

## Current Version

**V1 — Core Pipeline**, code complete, both ablation runs trained to
completion, and both now evaluated via `src/eval/evaluate.py`. What remains
is the cross-ablation write-up for the poster.

## Completion

**~100% of V1 execution.** Both ablation configs (`configs/gpu_server_internal.yaml`,
`configs/gpu_server_external.yaml`) have completed all 50 epochs and both
checkpoints have been scored on the CARD test holdout.

- External ablation (`configs/gpu_server_external.yaml`): **done and evaluated.**
  50/50 epochs, wandb run `cerulean-donkey-1`
  (https://wandb.ai/aidan-j-nguyen-san-jose-state-university/amr-soft-prompting/runs/snm1i141),
  checkpoint at `outputs/external/best_model.pt` (best val loss at epoch 44).
- Internal ablation (`configs/gpu_server_internal.yaml`): **done and evaluated.**
  50/50 epochs, wandb run `internal ablation v1`
  (https://wandb.ai/aidan-j-nguyen-san-jose-state-university/amr-soft-prompting/runs/zz67qllg),
  checkpoint at `outputs/internal/best_model.pt` (best val loss at epoch 49,
  the final epoch). Ran continuously 2026-07-11 00:59 → 2026-07-15 14:42
  (~4.57 days wall-clock, matching the earlier ~2.5-day-plus-contention
  estimate) with the gradient-checkpointing fix in place — no further OOMs.
- `src/eval/evaluate.py` has now been run against both checkpoints. Results
  written to `outputs/{internal,external}/eval/best_model_<timestamp>/`
  (JSON + confusion-matrix figures for resistance_mechanism and drug_class,
  aggregate + top-confused-pairs for the 398-class amr_gene_family task).

  | Metric | Internal (epoch 49) | External (epoch 44) |
  |---|---|---|
  | resistance_mechanism accuracy | 0.980 | 0.998 |
  | amr_gene_family accuracy | 0.835 | 0.873 |
  | amr_gene_family macro-F1 | 0.486 | 0.559 |
  | drug_class F1 (micro) | 0.913 | 0.999 |

  External injection outperforms internal on every metric, most sharply on
  drug_class F1. Not yet interpreted for the poster write-up — see What's Not
  Started.

## What's Working

Everything listed in the prior status entry still holds (`card_parser.py`,
`dataset.py` split/`AMRDataset`, `esm2_wrapper.py`, `soft_prompt.py`,
`classifier.py`, `loss.py`, `metrics.py`, `train.py`, `evaluate.py`,
`preprocess_card.py`, `run_training.py`, `load_config`, all five configs).
New this session:

- **Both ablation training runs are complete** — external finished last
  session; internal finished this session (50/50 epochs, no OOM) after
  running continuously since the gradient-checkpointing fix landed.
- **`src/eval/evaluate.py` run end-to-end for the first time**, against both
  checkpoints, on `spark-833c` (GPU server, per CLAUDE.md's escalation rule
  for 650M-model inference). Produces aggregate metrics, confusion matrices
  for resistance_mechanism/drug_class, and top-confused-pairs for
  amr_gene_family.
- `src/models/esm2_wrapper.py`'s internal-mode gradient-checkpointing
  approach was reworked from the manual `torch.utils.checkpoint.checkpoint`
  wrap (recorded in the last status entry) to HF's built-in
  `esm.gradient_checkpointing_enable()`, engaged by temporarily flipping each
  `EsmLayer.training` flag to `True` around the forward call (the model is
  otherwise kept in `eval()` throughout, per the frozen-ESM-2 constraint).
  Numerically inert either way — dropout is 0.0 for all four ESM-2 variants
  this project uses. New regression tests in `test_forward_pass.py`
  (`TestGradientCheckpointing`) guard that checkpointing is on only for
  internal mode and that the `.training` flag resets after `forward()`
  returns.
- `scripts/_check_external_wandb_state.py` — small helper (used by
  `scripts/check_memory_headroom.sh`) that prints the wandb state of the
  latest external-ablation run; not a user-facing entry point.

## What's In Progress

- Cross-ablation write-up (internal vs. external) for the poster, now
  unblocked — both runs trained and evaluated, results tabulated above.

## What's Not Started

1. **`src/data/prodigal_runner.py`** — nucleotide → AA translation. Still an
   empty stub, explicitly deferred to V2 by Aidan (2026-07-09): V1
   training/eval only ever consumes CARD's pre-translated amino acid FASTA,
   so this isn't needed for V1 functional completeness.
2. Poster write-up itself — interpreting *why* external injection
   outperforms internal (e.g. whether it's an artifact of the
   gradient-checkpointing path only internal mode needs, vs. a genuine
   architectural effect) and drafting the narrative/figures.
3. Deciding whether the ~4.5-day internal run time (vs. external's ~2.2 days)
   is worth investigating further or is an acceptable one-time training cost
   given V1 is inference/eval-bound going forward.

## Open Questions / Blockers

- **Is internal mode's clear underperformance (esp. drug_class F1: 0.913 vs
  0.999) a real architectural finding, or confounded by the
  gradient-checkpointing path?** Checkpointing changes memory/compute
  tradeoffs, not numerics, so it shouldn't explain the gap directly — but
  it's the one asymmetry between how the two runs trained, worth a sentence
  in the poster's limitations if the gap is highlighted as a finding.
- **`spark-833c` is a shared, unified-memory box** — GPU memory pool is the
  same pool as system RAM (`nvidia-smi` reports device-level `Memory-Usage`
  as "Not Supported" for this reason). Relevant if any further runs are
  launched on it; not currently blocking anything since both ablations are
  done.
- `docs/reviews/2026-07-05-classifier-loss-review.md` — remaining deferred
  items are low-severity doc/style nitpicks; unchanged.

## Recent Changes

1. **Internal ablation (`internal ablation v1`, zz67qllg) completed all 50
   epochs** — ran continuously 2026-07-11 00:59 → 2026-07-15 14:42
   (~4.57 days, wandb-reported runtime 394968s) with no further OOMs after
   the gradient-checkpointing fix. Checkpoint at `outputs/internal/best_model.pt`
   (epoch 49).
2. **`src/eval/evaluate.py` run for the first time**, against both
   `outputs/internal/best_model.pt` and `outputs/external/best_model.pt`, on
   `spark-833c`. Wrote JSON results + confusion-matrix figures to
   `outputs/{internal,external}/eval/`.
3. **First cross-ablation comparison available**: external injection beats
   internal on all three metrics tracked (resistance_mechanism accuracy,
   amr_gene_family accuracy/macro-F1, drug_class F1-micro), most sharply on
   drug_class F1 (0.999 vs 0.913).
4. **Gradient-checkpointing implementation reworked** from a manual
   `torch.utils.checkpoint.checkpoint` wrap to HF's built-in
   `gradient_checkpointing_enable()` plus a transient `EsmLayer.training`
   flag flip, since the model is kept in `eval()` throughout and the
   built-in switch only engages when a layer's own `.training` is `True`.
   Covered by new regression tests in `test_forward_pass.py`.
5. V1 execution is now essentially complete: both ablations trained and
   evaluated; remaining work is the poster write-up, not further code or
   training.
