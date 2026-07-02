# CLAUDE.md — amr-soft-prompting

This file configures Claude Code's behavior for this project. Read it fully before taking any action.

---

## Project Overview

This project applies **soft prompting to ESM-2** (a frozen protein language model) for **antimicrobial resistance (AMR) gene detection and classification**. The core novel contribution is the soft prompt module — specifically how biological metadata is encoded as numerical vectors that condition ESM-2 without updating its weights.

**Principal Investigator:** Professor Andreopolous (SJSU)
**Timeline:** 6-week summer research project
**Deliverable:** Research poster (publication TBD)

---

## Versioned Roadmap

The project is divided into three versions to prevent scope creep. **Respect these boundaries strictly.**

### V1 — Core Pipeline (current)
- ESM-2 soft prompting with CARD metadata (mechanism + drug class)
- Gene calling via Prodigal (nucleotide → amino acid)
- ESM-2 inference with mean-pooled embeddings
- Soft prompt module encoding CARD metadata as vectors
- MLP classification head
- Evaluation on CARD holdout set

### V2 — Extended Metadata (future)
- BLAST statistics (e-value, percent identity, alignment length)
- Phylogenetic metadata from RefSeq via MAFFT + iTOL
- Replaces coarse taxonomy with proper phylogenetic signals

### V3 — Knowledge Integration (future)
- TA loci proximity features from TADB 3.0
- KEGG functional annotation via KofamKOALA
- RAG (retrieval-augmented generation) as preprocessing enrichment
- Encodes qualitative text-based knowledge into soft prompt vectors

### Versioning Rules for Claude Code
- **Never implement V2 or V3 features while working on V1 tasks**
- If a task would naturally pull in V2/V3 scope, **stop and flag it explicitly** before proceeding
- Ask: "This touches V2 scope (BLAST/phylogeny). Should I proceed or stub it out?"
- Stubs and `# TODO: V2` comments are acceptable placeholders

---

## Architecture

```
Nucleotide FASTA
      │
  [Prodigal]           ← gene calling, nucleotide → amino acid
      │
Amino Acid Sequences
      │
  [ESM-2 frozen]       ← protein language model, weights never updated
      │
Mean-pooled Embeddings
      │                         ┌─────────────────────────┐
  [Soft Prompt Module] ←────────│ CARD Metadata Vectors   │
      │                         │ (mechanism, drug class)  │
      │                         └─────────────────────────┘
Combined Representation
      │
  [MLP Classification Head]
      │
AMR Gene Class Prediction
```

**Key constraint:** ESM-2 is always frozen. Gradients only flow through the soft prompt module and classification head.

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

**Rationale:** supervisor (Prof. Andreopolous) intends to continue this work with
other students for publication. Any researcher should be able to clone the repo,
run `preprocess_card.py`, and reproduce training exactly.

---

## Model Variants

| Environment     | Model              | Purpose                                      |
|-----------------|--------------------|----------------------------------------------|
| Local (WSL2)    | `esm2_t6_8M`       | Architecture iteration, unit tests, CI       |
| CPU server      | `esm2_t30_150M`    | Data pipeline validation, small inference    |
| GPU server (V1) | `esm2_t33_650M`    | All training runs                            |
| GPU server (V2+)| `esm2_t36_3B`      | If results justify scaling up                |

**Never hardcode model size or device.** Always read from the config file.

---

## Configuration System

All environment-specific settings live in `configs/`. Scripts must accept a `--config` argument and read from it. Never hardcode paths, device strings, model names, or hyperparameters in scripts.

```
configs/
  base.yaml          ← shared defaults (model architecture, training hyperparams)
  local.yaml         ← local WSL2 overrides (8M model, cpu, small batch)
  cpu_server.yaml    ← CPU server overrides (150M model, cpu, larger batch)
  gpu_server.yaml    ← GPU server overrides (650M model, cuda, full batch)
```

Example config structure:
```yaml
# configs/gpu_server.yaml
model:
  esm2_variant: "esm2_t33_650M_UR50D"
  device: "cuda"

training:
  batch_size: 32
  learning_rate: 1e-4
  epochs: 50
  freeze_esm2: true

paths:
  card_fasta: "/path/to/protein_fasta_protein_homolog_model.fasta"
  card_json: "/path/to/card.json"
  aro_index: "/path/to/aro_index.tsv"
  output_dir: "/path/to/outputs/"

logging:
  wandb_project: "amr-soft-prompting"
  wandb_run_name: null  # auto-generated if null
```

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
│   └── gpu_server.yaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── card_parser.py       ← parse CARD FASTA + ARO index + card.json
│   │   ├── prodigal_runner.py   ← wrap Prodigal for nucleotide → AA translation
│   │   └── dataset.py           ← PyTorch Dataset class for AMR sequences
│   ├── models/
│   │   ├── __init__.py
│   │   ├── esm2_wrapper.py      ← load frozen ESM-2, extract embeddings
│   │   ├── soft_prompt.py       ← soft prompt module (the core contribution)
│   │   └── classifier.py        ← MLP classification head
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py             ← main training loop
│   │   └── loss.py              ← loss functions
│   └── eval/
│       ├── __init__.py
│       └── evaluate.py          ← metrics, confusion matrix, per-class breakdown
├── scripts/
│   ├── preprocess_card.py       ← one-time data preparation
│   └── run_training.py          ← entry point: python scripts/run_training.py --config configs/gpu_server.yaml
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

### TADB 3.0 — V3 only
- Toxin and antitoxin FASTA files
- **Do not integrate until V3 is explicitly started**

### RefSeq — V2 only
- Microbial genomes for negative training examples and phylogenetic metadata
- Organism scope to be confirmed with Andreopolous
- **Do not integrate until V2 is explicitly started**

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

**On the GPU server (SSH session):**
- Run all training jobs: `python scripts/run_training.py --config configs/gpu_server.yaml`
- Run evaluation scripts
- Monitor wandb outputs and fix errors in place
- Use config: `configs/gpu_server.yaml`

**GitHub (handoff between environments):**
- Push from whichever environment code was written in
- Pull on the destination environment before running
- Never commit: `outputs/`, checkpoints, wandb cache, `.pyc` files

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
> "This job requires the GPU server (650M model / training run). Please SSH into the GPU server and re-run this command with `--config configs/gpu_server.yaml`."

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
- **RAG is V3 and was Andreopolous's suggestion.** Do not add RAG components until V3 is explicitly started.
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
