# amr-soft-prompting

Soft prompting on a frozen ESM-2 protein language model for antimicrobial
resistance (AMR) gene classification. Three single-head tasks, each ablated on
soft-prompt injection mode (internal vs. external). See `CLAUDE.md` for the
full spec and `docs/STATUS.md` for current project state.

## Setup (Linux / WSL2)

```bash
git clone https://github.com/ajnguyen14/amr-soft-prompting.git ~/amr-soft-prompting
cd ~/amr-soft-prompting
conda create -n amr-esm2 python=3.11 -y
conda activate amr-esm2
pip install -r requirements.txt
wandb login
pytest tests/ -v          # smoke tests, 8M model, CPU
```

Clone into the Linux filesystem (`~/`), not `/mnt/c/...`.

## Data

Raw data and processed artifacts are gitignored. Place the CARD v4.0.1 files
(`protein_fasta_protein_homolog_model.fasta`, `card.json`, `aro_index.tsv`) in
`data/raw/`, then run:

```bash
python scripts/preprocess_card.py --config configs/local.yaml
```

The TA-proximity pipeline (`scripts/run_blast_coordinate_mapping.py`,
`scripts/run_ta_proximity.py`) additionally needs NCBI BLAST+ on the PATH and
the TADB 3.0 type II FASTA files.

## Training

```bash
python scripts/run_training.py --config configs/gpu_task3_genefamily_internal.yaml
```

Task configs live in `configs/` (`gpu_task{1,2,3}_*_{internal,external}.yaml`,
plus `*_noconditioning_*` negative controls). Internal/external pairs differ
only in `injection_mode`. Run on a GPU server; use `configs/local.yaml` for
CPU-only checks.

## Layout

`src/` library code, `scripts/` entry points, `configs/` run specs,
`tests/` smoke tests, `docs/` status, plans, poster and `archive/` session logs,
`notebooks/` analysis notebooks.
