# Session Log — 2026-06-29

## Summary

Full V1 data pipeline session. Went from an empty repo (just a README and .gitignore) to a working
CARD data parser with 22/22 passing tests. Also established the project scaffold, config system,
and comprehensive `CLAUDE.md` operating instructions.

## Commits

| Hash | Message |
|------|---------|
| `5e263b4` | Created project skeleton (24 empty files) |
| `197b385` | Update requirements.txt |
| `6bbaca0` | Create CLAUDE.md |
| `addca65` | feat: V1 data pipeline — card_parser, CARDRecord, label vocabularies (22/22 passing) |

## What Was Done

### Project Scaffold (`5e263b4`)
Created the full directory tree with empty placeholder files:
- `configs/` — `base.yaml`, `local.yaml`, `cpu_server.yaml`, `gpu_server.yaml`
- `src/data/` — `card_parser.py`, `dataset.py`, `prodigal_runner.py`
- `src/models/` — `esm2_wrapper.py`, `soft_prompt.py`, `classifier.py`
- `src/training/` — `train.py`, `loss.py`
- `src/eval/` — `evaluate.py`
- `scripts/` — `preprocess_card.py`, `run_training.py`
- `tests/` — `test_data_pipeline.py`, `test_soft_prompt.py`, `test_forward_pass.py`

### Requirements (`197b385`)
Pinned initial dependencies in `requirements.txt`: torch, fair-esm, biopython, pyyaml, wandb,
pytest, numpy, pandas, scikit-learn.

### CLAUDE.md (`6bbaca0`)
Wrote comprehensive project-level instructions covering:
- Versioned roadmap (V1/V2/V3) with explicit scope boundaries
- Architecture diagram (Prodigal → ESM-2 frozen → soft prompt → MLP head)
- Model variant table by environment (8M local / 150M CPU / 650M GPU)
- Config system spec and example yaml structure
- Three-environment workflow (WSL2 → CPU server → GPU server)
- Server escalation rules (what to run where, when to refuse)
- Code style, testing philosophy, experiment tracking (wandb) conventions

### Card Parser & Tests (`addca65`)
**`src/data/card_parser.py`** — core V1 data ingestion:
- `CARDRecord` dataclass with all V1 metadata fields (ARO accession, protein accession,
  gene name, organism, sequence, drug classes, resistance mechanism, AMR gene family,
  CARD short name).
- `_parse_fasta_header()` — regex-based parser for CARD FASTA description lines
  (`gb|<acc>|ARO:<id>|<gene_name> [<organism>]`).
- `_parse_aro_index()` — loads `aro_index.tsv` into an ARO-keyed dict.
- `load_card_dataset()` — joins FASTA sequences with ARO index metadata; handles
  multi-drug-class semicolon splitting; accepts optional `card_json_path` as a
  `# TODO: V2` forward-compat stub.
- `get_label_vocabularies()` — builds sorted, deduplicated vocabularies for
  `drug_class`, `resistance_mechanism`, and `amr_gene_family` — ready for integer
  encoding by `AMRDataset`.

**`tests/test_data_pipeline.py`** — 22 tests across four classes:
- `TestParseFastaHeader` — standard header, gene name with parentheses, malformed raises.
- `TestMinimalDataset` — record count, types, ARO string format, non-empty sequences,
  multi-drug-class splitting, metadata alignment.
- `TestGetLabelVocabularies` — required keys, sorted order, no duplicates, multi-class
  expansion.
- `TestFullCARDDataset` — guarded by `pytest.mark.skipif` when CARD data absent;
  asserts 6052 records, uniqueness, no missing mechanism/drug class.

**Supporting files:**
- `conftest.py` — adds project root to `sys.path` so `src.*` imports resolve under pytest.
- `pyproject.toml` — sets `testpaths = ["tests"]` and `pythonpath = ["."]` for pytest.
- `configs/gpu_server.yaml` — filled in absolute paths to CARD data on the CPU/GPU server
  (`/data/aidannguyen/amr-soft-prompting/data/raw/`).

## Key Design Decisions

- **Multi-hot drug class encoding** decided here conceptually. 57% of CARD records carry
  more than one drug class (distribution: 2589×1, 1482×2, 1593×3, 308×4, tail to 14),
  so multi-label is the default case, not an edge case.
- **`card.json` accepted but unused in V1.** `load_card_dataset` takes it as an optional
  parameter to avoid a future breaking change, but all V1 metadata comes from
  `aro_index.tsv`. A `# TODO: V2` comment marks the extension point.
- **Tests run on in-memory fixtures.** The minimal FASTA + TSV fixtures are embedded as
  strings so tests pass instantly anywhere without the CARD data files.

## State at Session End

- `src/data/card_parser.py` — complete, 22/22 tests passing.
- `src/data/dataset.py` — empty placeholder (not yet implemented).
- `src/data/prodigal_runner.py` — empty placeholder.
- All model and training files — empty placeholders.

## Open Questions / Next Steps

- Implement `AMRDataset` in `dataset.py` with multi-hot drug class and integer
  single-label encoding.
- Add smoke tests for `AMRDataset` (`tests/test_dataset.py`).
- Implement `esm2_wrapper.py` (load frozen ESM-2, extract mean-pooled embeddings).
- Implement `soft_prompt.py` (the core novel contribution).
