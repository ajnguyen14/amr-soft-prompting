# CLAUDE.md — amr-soft-prompting

This file configures Claude Code's behavior for this project. Read it fully before taking any action.

---

## Project Overview

This project applies **soft prompting to ESM-2** (a frozen protein language model) for **antimicrobial resistance (AMR) gene classification**. The core novel contribution is the soft prompt module — specifically how biological metadata is encoded as numerical vectors that condition ESM-2 without updating its weights.

**As of 2026-07-23, the project runs three separate single-head classification tasks** (not one multi-head classifier — see "Single-Head Architecture" below), each independently ablated across soft-prompt injection mode (internal vs. external).

**Principal Investigator:** Professor Andreopoulos (SJSU)
**Timeline:** 6-week summer research project
**Deliverable:** Research poster (Aug 7 deadline; publication TBD)

---

## Versioned Roadmap

The project is divided into versions to prevent scope creep. **Respect these boundaries strictly.**

**Scope note (2026-07-23, per Andreopoulos):** TA loci (TADB 3.0) proximity is
now part of **V2**, paired with RefSeq mapping, as a single cohesive unit —
this supersedes the original split where TA loci were V3-only. RefSeq's role
also changed: it is no longer for negative training examples or
phylogenetics, but for **BLAST-mapping CARD sequences onto genomic
coordinates** so TA-locus bp distance can be computed. See "TA-Proximity
Pipeline" below for the full spec.

### V1 — Core Pipeline (complete)
- ESM-2 soft prompting with CARD metadata (mechanism + drug class)
- ESM-2 inference with mean-pooled embeddings, frozen backbone
- Soft prompt module encoding CARD metadata as vectors
- Single-head MLP classifier on `amr_gene_family` (see "Label Leakage" below
  for why this is the only V1 target)
- Internal vs. external injection ablation
- Evaluation on CARD holdout set
- **Status:** code complete, both ablations retrained post-leakage-fix
  (`i7o4eg5n` internal, `2rr2h1f9` external — 0.9087 / 0.9054 final val gene
  family accuracy). Gene calling via Prodigal was scoped out of V1 entirely
  (CARD's pre-translated amino acid FASTA is used directly) and remains an
  empty stub, deferred to a later version if ever needed.

### V2 — Single-Head Retargeting + TA-Proximity Conditioning (current)
Two changes, both specified by Andreopoulos (2026-07-21 meeting):

1. **Single-head classifiers, one prediction target per run** — replacing the
   old idea of one classifier with multiple simultaneous heads. See
   "Single-Head Architecture" below for the full task/conditioning matrix.
2. **TA loci proximity (TADB 3.0) as a new soft-prompt conditioning input**,
   specifically for the `amr_gene_family` prediction task. Requires mapping
   CARD sequences onto RefSeq genomic coordinates via BLAST, then computing
   bp distance to the nearest TA locus on the same replicon. See
   "TA-Proximity Pipeline" below.

### V3 — Knowledge Integration (future, not yet scoped in detail)
- KEGG functional annotation via KofamKOALA
- RAG (retrieval-augmented generation) as preprocessing enrichment
- Encodes qualitative text-based knowledge into soft prompt vectors
- Binary AMR detection (originally Andreopoulos's idea) is a candidate
  direction here but is **not currently scoped** — would require RefSeq
  negative sampling design (easy vs. hard negatives) that hasn't been
  started. Do not build toward this without an explicit go-ahead.

### Versioning Rules for Claude Code
- **Never implement V3 features while working on V1/V2 tasks**
- If a task would naturally pull in V3 scope (KEGG, RAG, binary detection),
  **stop and flag it explicitly** before proceeding
- Stubs and `# TODO: V3` comments are acceptable placeholders

---

## Architecture

```
Amino Acid Sequences (from CARD's pre-translated FASTA)
      │
  [ESM-2 frozen]       ← protein language model, weights never updated
      │
Mean-pooled Embeddings
      │                         ┌───────────────────────────┐
  [Soft Prompt Module] ←────────│ Conditioning input        │
      │                         │ (task-dependent — see      │
      │                         │  Single-Head Architecture) │
      │                         └───────────────────────────┘
Combined Representation
      │
  [Single-Head Classifier]     ← exactly one prediction target per run
      │
Prediction (task-dependent)
```

**Key constraint:** ESM-2 is always frozen. Gradients only flow through the soft prompt module and classification head.

### Label Leakage — Why the Architecture Changed (2026-07-17)

V1 originally scored `resistance_mechanism` and `drug_class` as classifier
targets *while also* feeding their ground-truth values into the soft prompt
as conditioning input. This let those two heads shortcut — decoding their
own conditioning input rather than learning anything from the ESM-2 sequence
representation (observed as near-ceiling accuracy from epoch 1, a leakage
signature, not a real result). Fixed by dropping both as prediction targets,
leaving `amr_gene_family` — never fed into the soft prompt — as V1's only
honest metric.

**This is the reason single-head retargeting (V2) is structured the way it
is below**: every task/conditioning pairing is chosen so the conditioning
input is never also the prediction target for that run. Apply the same
scrutiny to any new task before implementing it — if a proposed conditioning
input could plausibly leak into a proposed target, flag it and check
empirically (does the conditioning input alone predict the target, with no
ESM-2 involved?) before trusting results.

### Single-Head Architecture (V2, per Andreopoulos 2026-07-21)

Three separate training runs, each with exactly one classifier head and one
soft-prompt conditioning input. **Do not combine multiple targets into one
classifier** — this was the source of the original leakage and, independent
of leakage, multi-task heads sharing one trunk can suffer task interference.

| Run | Conditioning input (soft prompt) | Prediction target | Loss |
|---|---|---|---|
| 1 | `amr_gene_family` | `drug_class` | `BCEWithLogitsLoss` (multi-label) |
| 2 | `amr_gene_family` | `resistance_mechanism` | `CrossEntropyLoss` |
| 3 | TA-proximity (see below) | `amr_gene_family` | `CrossEntropyLoss` |

Each of the three runs is independently ablated on injection mode
(internal vs. external) per Andreopoulos's direction — **6 total training
runs**. Injection-mode ablation logic itself (in `esm2_wrapper.py`) is
target-agnostic and unchanged by this restructuring.

### TA-Proximity Pipeline (V2, Run 3's conditioning input)

Computes a categorical TA-locus-proximity feature per CARD ARO accession,
used only as Run 3's soft-prompt conditioning input (see table above).

1. **BLAST CARD proteins against RefSeq** to place each ARO accession at a
   genomic coordinate (`replicon_accession:start-end`). Threshold choices
   (e-value, %identity, coverage) are not yet finalized — pick conservative
   defaults (e.g. ≥95% identity) and record the actual coverage achieved
   (% of ARO accessions successfully mapped) as a reportable number, not
   just a pipeline detail.
2. **Parse TADB 3.0 FASTA headers** (`type_II_{T,AT}_{exp,pre}.fas`) for
   `replicon:coords` — already RefSeq-anchored, no BLAST needed for this
   side. Use both `_exp` (403 pairs, high confidence, sparse) and `_pre`
   (larger, lower confidence) for usable coverage.
3. **Same-replicon bp distance**: only compute a distance when the CARD hit
   and a TADB locus share the exact same RefSeq replicon accession — never
   compare coordinates across different replicons/assemblies, even from the
   same organism. Take distance to the *nearest* TA locus on that replicon.
4. **Encode as a categorical embedding** (`nn.Embedding`, matching the
   existing pattern for `mechanism`/`drug_class` — do NOT use a raw or
   log-scaled continuous scalar projection; a single linear direction can't
   represent proximity's nonlinear relevance and forces missing-value
   sentinels onto the same axis as real distances). Vocabulary:
   - `unknown` — ARO accession did not map to RefSeq (Step 1 failed).
     This is a **data-quality gap**, not a biological signal — never encode
     it as a distance value.
   - `no_ta_locus` — mapped successfully, but no TA locus exists on that
     replicon. This **is** a real biological signal, distinct from
     `unknown` — do not conflate the two, and do not encode as a large
     distance number.
   - One embedding entry per **distance bin** for real same-replicon
     distances. Bin edges are not yet finalized — set them from the actual
     distance histogram once Steps 1–3 are run, not chosen a priori.

---

## Reproducibility Requirements

- **Random seeds** — set explicitly everywhere: PyTorch, numpy, and Python's `random`
  module. Use seed `42` as the project default.
- **Train/val/test split** — always split on ARO accessions, never sequences.
  Stratified by resistance mechanism. 80/10/10 ratio. Split must be deterministic
  given the fixed seed.
- **Preprocessing entry point** — `scripts/preprocess_card.py` must be a single
  runnable script that takes raw CARD files and produces all data artifacts needed
  for training. No manual steps outside this script.
- **Config completeness** — every training run must be fully specified by a config
  file. No hardcoded hyperparameters in source code.
- **Experiment logging** — all training runs logged to wandb with the full config
  that produced them. A result without a corresponding config is not reproducible.

**Rationale:** supervisor (Prof. Andreopoulos) intends to continue this work with
other students for publication. Any researcher should be able to clone the repo,
run `preprocess_card.py`, and reproduce training exactly.

---

## Model Variants

| Environment     | Model              | Purpose                                      |
|-----------------|--------------------|----------------------------------------------|
| Local (WSL2)    | `esm2_t6_8M`       | Architecture iteration, unit tests, CI       |
| CPU server      | `esm2_t30_150M`    | Data pipeline validation, small inference    |
| GPU servers     | `esm2_t33_650M`    | All training runs (see below — 3 servers)    |
| GPU server (future) | `esm2_t36_3B`  | If results justify scaling up                |

**Never hardcode model size or device.** Always read from the config file.

**GPU server is no longer a single machine.** There are 3 physically separate
GPU-capable servers: DGX Spark (`spark-833c`, NVIDIA GB10, large unified
memory), and two independent RTX 3090 boxes (one is `sjsu`). Confirmed from
wandb run metadata: **the 3090s are meaningfully faster per-epoch for this
workload than the DGX Spark** (post-fix V1 runs on `sjsu`: ~16 min/epoch
internal, ~5.4 min/epoch external, at batch 24 — vs. pre-fix runs on
`spark-833c`: ~152 min/epoch internal, ~25 min/epoch external, at batch 32).
This was a hardware effect, not a code regression — don't assume Spark is
the "primary" or fastest server by default. With 4 total GPUs across 3
machines, the 6 single-head ablation runs (see Single-Head Architecture)
can run substantially in parallel rather than queued serially — check
`wait_and_run.sh` usage per-server before launching concurrent jobs, since
shared-memory contention is a known issue on the Spark box specifically.

---

## Configuration System

All environment-specific settings live in `configs/`. Scripts must accept a `--config` argument and read from it. Never hardcode paths, device strings, model names, or hyperparameters in scripts.

```
configs/
  base.yaml                       ← shared defaults (model architecture, training hyperparams)
  local.yaml                      ← local WSL2 overrides (8M model, cpu, small batch)
  cpu_server.yaml                 ← CPU server overrides (150M model, cpu, larger batch)
  gpu_task1_drugclass_internal.yaml    ← Run 1 (drug_class ← gene_family), internal injection
  gpu_task1_drugclass_external.yaml    ← Run 1 (drug_class ← gene_family), external injection
  gpu_task2_mechanism_internal.yaml    ← Run 2 (resistance_mechanism ← gene_family), internal
  gpu_task2_mechanism_external.yaml    ← Run 2 (resistance_mechanism ← gene_family), external
  gpu_task3_genefamily_internal.yaml   ← Run 3 (gene_family ← TA-proximity), internal
  gpu_task3_genefamily_external.yaml   ← Run 3 (gene_family ← TA-proximity), external
```

(Filenames above are illustrative — match actual naming convention when
creating these; the old `gpu_server_internal.yaml`/`_external.yaml` pair
should be treated as V1 legacy configs and not deleted, since the retrained
checkpoints from those files are current V1 results.)

**Within each task**, the internal/external config pair must be identical in
every field except `injection_mode` — same parity rule as V1. **Any
hyperparameter change (learning rate, batch size, epochs, optimizer, etc.)
must be applied to both files in the pair**, or the ablation is no longer a
controlled A/B test.

**Cross-server note:** the project now runs across 3 physical GPU-capable
servers (DGX Spark, and two separate RTX 3090 machines — see Workflow
section). Batch size is **not** necessarily uniform across all 6 configs —
the 3090s have a lower memory ceiling than initially assumed (confirmed
batch 24 is the practical max on `sjsu`, vs. batch 32 used on `spark-833c`).
If a task's internal/external pair both run on the same server, keep batch
size identical between them (parity rule above still applies within a pair).
If different servers genuinely require different batch sizes, that's an
intentional divergence — document it with a comment in the relevant config
files rather than leaving it to look like an inconsistency.

Example config structure:
```yaml
# example: configs/gpu_task3_genefamily_internal.yaml
model:
  esm2_variant: "esm2_t33_650M_UR50D"
  device: "cuda"
  injection_mode: "internal"   # the only field that differs from the paired _external.yaml

training:
  batch_size: 32
  learning_rate: 1.0e-4   # YAML gotcha: "1e-4" (no decimal point) parses as a
                          # string in PyYAML, not a float — always write
                          # "1.0e-4" or "0.0001"
  epochs: 50
  optimizer: "adam"

paths:
  card_fasta: "/path/to/protein_fasta_protein_homolog_model.fasta"
  card_json: "/path/to/card.json"
  aro_index: "/path/to/aro_index.tsv"
  output_dir: "/path/to/outputs/"

logging:
  wandb_project: "amr-soft-prompting"
  wandb_run_name: null  # auto-generated if null
```

Config loading: `src/utils/config.py`'s `load_config(config_path)` reads
`config_path` and deep-merges it over `configs/base.yaml` (nested sections merge
key-by-key, not wholesale replacement), so an environment file only needs to
specify what it actually overrides.

---

## Project Structure

```
amr-soft-prompting/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── local.yaml
│   ├── cpu_server.yaml
│   ├── gpu_task1_drugclass_internal.yaml
│   ├── gpu_task1_drugclass_external.yaml
│   ├── gpu_task2_mechanism_internal.yaml
│   ├── gpu_task2_mechanism_external.yaml
│   ├── gpu_task3_genefamily_internal.yaml
│   └── gpu_task3_genefamily_external.yaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── card_parser.py       ← parse CARD FASTA + ARO index + card.json
│   │   ├── prodigal_runner.py   ← wrap Prodigal for nucleotide → AA translation
│   │   └── dataset.py           ← PyTorch Dataset class + train/val/test split
│   ├── models/
│   │   ├── __init__.py
│   │   ├── esm2_wrapper.py      ← load frozen ESM-2, extract embeddings
│   │   ├── soft_prompt.py       ← soft prompt module (the core contribution)
│   │   └── classifier.py        ← MLP classification head
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py             ← main training loop
│   │   └── loss.py              ← loss functions
│   ├── eval/
│   │   ├── __init__.py
│   │   └── evaluate.py          ← metrics, confusion matrix, per-class breakdown
│   └── utils/
│       ├── __init__.py
│       └── config.py            ← load_config: merges base.yaml + environment override
├── scripts/
│   ├── preprocess_card.py       ← one-time data preparation
│   └── run_training.py          ← entry point: python scripts/run_training.py --config <task config, see configs/>
├── tests/
│   ├── test_data_pipeline.py    ← smoke tests for data loading and label alignment
│   ├── test_soft_prompt.py      ← smoke tests for soft prompt module shapes
│   └── test_forward_pass.py     ← end-to-end forward pass sanity check
├── notebooks/
│   └── exploration/             ← scratch notebooks, never imported by src/
└── outputs/                     ← gitignored; holds checkpoints, logs, wandb cache
```

---

## Datasets

### CARD (broadstreet v4.0.1) — V1
- `protein_fasta_protein_homolog_model.fasta` — amino acid sequences
- `card.json` — full metadata including mechanism and drug class
- `aro_index.tsv` — ARO accession mappings

### TADB 3.0 — V2, in progress
- Type II toxin/antitoxin protein FASTA, both `_exp` (experimentally
  validated, 403 pairs) and `_pre` (in silico predicted, much larger)
- Coordinates embedded directly in FASTA headers
  (`>ID accession replicon:coords [organism]`) — no need to run TAfinder
  locally, headers are already RefSeq-anchored
- Regulator protein files (Type II only, exp-only, no `_pre` counterpart)
  are **not used** — redundant with toxin/antitoxin coordinates for the
  same locus, adds parsing complexity for no new proximity information
- See "TA-Proximity Pipeline" above for how this is used

### RefSeq — V2, in progress
- Used to BLAST-map CARD protein sequences onto genomic coordinates, so
  TA-locus bp distance can be computed (see "TA-Proximity Pipeline" above)
- **Not** used for negative training examples or phylogenetics — that was
  an earlier, superseded scoping of V2's purpose
- Representative genomes subset; exact organism scope still being
  finalized as the BLAST step is built out

---

## Dependencies & Environment

- **Conda environment:** `amr-esm2` (Python 3.11)
- **Dependency tracking:** `requirements.txt` — keep it updated whenever a new package is installed
- **Install command:** `pip install -r requirements.txt` (always inside `amr-esm2`)
- **Never suggest `conda install` for new packages** — use pip for consistency
- When adding a new dependency, add it to `requirements.txt` immediately with a pinned version

Key packages (reference):
```
torch
fair-esm
transformers
biopython
pyyaml
wandb
pytest
numpy
pandas
scikit-learn
```

---

## Experiment Tracking

All training and evaluation scripts use **Weights & Biases (wandb)**.

- Project name: `amr-soft-prompting`
- Log at minimum: loss, accuracy, F1 per class, learning rate, epoch, config snapshot
- Always log the full config dict at run start: `wandb.config.update(config)`
- Run names should be descriptive: `v1-650M-gpu-lr1e4-epoch50`
- Never disable wandb logging without an explicit comment explaining why

---

## Code Style

- **Comments:** moderate — docstrings on all functions and classes, inline comments on non-obvious logic only
- **Type hints:** always include on function signatures
- **Docstring format:** Google style
- **Line length:** 100 characters max
- **No magic numbers:** all constants go in config or as named variables with a comment

Example:
```python
def encode_metadata(
    mechanism: str,
    drug_class: str,
    embed_dim: int,
) -> torch.Tensor:
    """Encode CARD metadata fields into a soft prompt vector.

    Args:
        mechanism: AMR resistance mechanism string from CARD (e.g. 'antibiotic efflux').
        drug_class: Drug class string from CARD (e.g. 'fluoroquinolone antibiotic').
        embed_dim: Dimensionality of the output embedding vector.

    Returns:
        Tensor of shape (embed_dim,) representing the encoded metadata.
    """
```

---

## Testing

- **Framework:** pytest
- **Philosophy:** lightweight smoke tests — assert shapes, types, and label alignment; not exhaustive coverage
- **Run on CPU server with:** `pytest tests/ -v`
- Every new module in `src/` should have a corresponding smoke test in `tests/`
- Smoke tests must use the **8M model** so they complete in seconds — never the 650M model
- Claude Code should run the relevant smoke test automatically after writing or editing any module in `src/`

Minimum smoke test checklist:
- [ ] Data loader returns correct tensor shapes
- [ ] Labels map correctly to ARO accessions
- [ ] Soft prompt module output shape matches ESM-2 input expectations
- [ ] Full forward pass completes without error (8M model, batch size 2, CPU)

---

## Workflow

Claude Code runs directly on the servers via SSH from a Surface Pro. GitHub is the handoff mechanism between environments — not the primary execution context.

### Three-Environment Model

```
Local WSL2                  CPU Server                    GPU Server
──────────────────          ──────────────────────        ──────────────────────
Architecture design    →    Data pipeline scripts    →    Training runs
Config drafting             Preprocessing (CARD,           Evaluation
Quick logic checks          Prodigal)                      Full inference
                            Smoke tests                    Hyperparameter sweeps
                            Pipeline validation
                            Small inference tests
```

### How Claude Code operates per environment

**On the CPU server (SSH session):**
- Run data preprocessing, CARD parsing, Prodigal wrapping
- Run all smoke tests: `pytest tests/ -v`
- Run small inference tests with 150M model to validate pipeline shapes
- Fix errors and bugs in place — Claude Code sees outputs directly
- Use config: `configs/cpu_server.yaml`

**On a GPU server (SSH session — DGX Spark or either RTX 3090 box):**
- Run all training jobs: `python scripts/run_training.py --config <task-specific config>`
  — see Configuration System above for the 6 current single-head task configs
- Run evaluation scripts
- Monitor wandb outputs and fix errors in place; wandb run metadata records
  hostname/GPU automatically — useful for debugging cross-server timing
  differences (see Model Variants above)
- Check for concurrent jobs on the same server before launching a new one —
  especially on the DGX Spark, where GPU memory is a shared unified pool
  with system RAM (`wait_and_run.sh` exists for this)

**GitHub (handoff between environments):**
- Push from whichever environment code was written in
- Pull on the destination environment before running
- Never commit: `outputs/`, checkpoints, wandb cache, `.pyc` files

### Identify the Current Server Before Doing Anything Else

At the start of every session, before running any job, **check which
environment you're actually on** — don't assume from what the person says or
from what the last session was. Run something like:

```bash
hostname
nvidia-smi -L 2>/dev/null || echo "no GPU detected"
```

Match the result against known hosts:

| `hostname` output | Environment | Capabilities |
|---|---|---|
| (local WSL2, no fixed hostname convention) | Local | 8M model only, CPU |
| CPU server hostname | CPU server | 150M model, CPU, batch ≤ 8 |
| `spark-833c` | DGX Spark (GPU) | 650M model, GPU, batch 32 confirmed working, unified-memory pool shared with system RAM — check `wait_and_run.sh` before launching |
| `sjsu` (or other RTX 3090 hostname) | RTX 3090 box (GPU) | 650M model, GPU, batch 24 is the confirmed practical ceiling (not 32 — see Model Variants above) |

If the hostname doesn't match anything in this table (e.g. a new server was
added, or a hostname was renamed), **stop and ask** rather than guessing
which capabilities apply — don't assume it's CPU-only or GPU-capable without
confirming.

This matters because the three GPU-capable servers are not interchangeable:
they differ in per-epoch speed (see Model Variants above — the 3090s are
substantially faster than Spark for this workload) and in the batch size
they can actually sustain. A command that's safe on one GPU server (e.g.
batch 32) may OOM on another (the 3090s cap at 24). Knowing which server
you're on is a prerequisite for knowing which config file and which
capability limits apply for the rest of the session — not just for the
CPU-vs-GPU decision in the escalation rules below.

### Server Escalation Rules

Claude Code must check the current environment before running any job. If the job exceeds CPU server limits, **stop and say so explicitly** rather than attempting to run it.

| Condition | Environment | Action |
|---|---|---|
| Smoke tests, shape checks, pipeline validation | CPU server | ✅ Run here |
| Preprocessing scripts (CARD, Prodigal) | CPU server | ✅ Run here |
| Inference with 8M or 150M model, batch ≤ 8 | CPU server | ✅ Run here |
| Inference with 650M+ model | GPU server | 🚫 Do not run on CPU server |
| Any training run with gradient updates | GPU server | 🚫 Do not run on CPU server |
| Full dataset inference (>1000 sequences) | GPU server | 🚫 Do not run on CPU server |

If Claude Code is on the CPU server and asked to do something in the 🚫 column, respond:
> "This job requires a GPU server (650M model / training run). Please SSH into one of the GPU servers (DGX Spark or an RTX 3090 box) and re-run this command with the relevant task config from `configs/`."


### General Rules
- **Never commit** `outputs/`, checkpoints, or wandb cache — gitignored
- **Always commit** `configs/` — configs are part of the reproducible record
- **Commit `requirements.txt`** changes alongside the code that requires them
- **Always activate** `conda activate amr-esm2` before running any script

---

## Boundaries & Constraints

- **ESM-2 is always frozen.** Never set `requires_grad=True` on ESM-2 parameters.
- **The soft prompt module is the novel contribution.** Treat it with the most care — document design decisions, flag any architectural changes before implementing.
- **Prodigal handles nucleotide → amino acid translation.** ESM-2 never sees raw nucleotide sequences.
- **RAG is V3 and was Andreopoulos's suggestion.** Do not add RAG components until V3 is explicitly started.
- **AlphaFold is out of scope** for all versions of this project.
- **KEGG is V3 only.** Do not scaffold KEGG integration in V1 or V2.

---

## When in Doubt

1. Check the versioned roadmap — is this V1 scope?
2. Check the architecture diagram — does this fit the established flow?
3. Flag it and ask rather than assuming and building.

## Project Status File

At the end of every session, also overwrite `docs/STATUS.md` with the current project state. Use this template:

# AMR Soft Prompting — Project Status
_Last updated: YYYY-MM-DD HH:MM_

## Current Version
Which version is actively being developed (V1 / V2 / V3) and what stage it's at.

## Completion
High-level percentage estimate and what that's based on.

## What's Working
- Bullet list of completed, tested components.

## What's In Progress
- Bullet list of components currently being built.

## What's Not Started
- Bullet list of remaining components, in intended build order.

## Open Questions / Blockers
- Anything unresolved that affects next steps.

## Recent Changes
The 3–5 most impactful changes from the last session, in plain English.