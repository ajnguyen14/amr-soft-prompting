# V3 Plan — AMR Detection + Property Prediction

> **Status: PROPOSED — pending confirmation from Prof. Andreopoulos (drafted 2026-10-08).**
> Negative sampling (Part A) and cross-validation (Part B) are approved and can proceed now.
> Training the binary detector waits for confirmation of this rescope and for Bill's scoping
> of the task shape.

## 1. Motivation

Andreopoulos wants the project to become a tool that helps students learn about antimicrobial
resistance. The proposed V3 changes the project's main question from *"which family or class
is this known AMR protein?"* to *"which proteins in this genome look like AMR genes, and what
are they likely to do?"*

This replaces the earlier V3 scope (KEGG/KofamKOALA, RAG). Those move to **V4 / unscheduled**.

### Ownership

| Component | Owner |
|---|---|
| Data (negatives, matched positives, genome context), model training, cross-validation, benchmark, bioRxiv paper | Aidan (this repo) |
| Student-facing interface | Another student (separate project; consumes the inference contract in §7) |

The **bioRxiv paper covers the model and benchmark only.** The interface is a follow-on deliverable
and does not block the preprint.

## 2. Pipeline

```
Genome (nucleotide FASTA)            ← tool input; benchmark uses RefSeq assemblies directly
      │
  [Gene calling: Prodigal]           ← src/data/prodigal_runner.py (stub → back in scope)
      │
Proteins + coordinates + genome context (replicon type, TA-proximity, MGE neighbours)
      │
  [ESM-2 frozen] + [soft prompt ← genome context]
      │
  Stage 1: binary AMR-like detector        → p_amr per protein (calibrated)
      │  (proteins above threshold)
  Stage 2: property heads                   → gene family / drug class / mechanism
      │
Highlighted AMR-like proteins, each with predicted properties, confidence and nearest CARD hit
```

## 3. Conditioning: genome context, not ground-truth labels

V2's Runs 1 and 2 condition on the **true** `amr_gene_family`. A protein scanned from a
student's genome has no such label, so those models can't be deployed as they are. In V3 every
head is conditioned on **genome context** that exists for any protein in an annotated or
Prodigal-called genome:

| Feature | Values | Notes |
|---|---|---|
| `replicon_type` | chromosome / plasmid | Real biology: AMR genes are enriched on plasmids |
| `ta_proximity` | `distance` / `no_ta_locus` / `unknown` | Same 3-way vocab as V2 Run 3, computed from native coordinates (no BLAST mapping) |
| `mge_neighbor` | yes / no | Transposase, integrase or recombinase within N kb (N set in config) |

**Leakage gate (required before any feature is used):** score each feature alone (the
`conditioning_lookup` baseline in `src/eval/baselines.py`) on AMR vs. non-AMR. If it separates
the classes because of real biology, it may be used, but its score becomes a baseline the model
must beat. If it separates them only because of how the data was built, drop it. A "non-AMR"
token, or any value that only negatives can take, is never allowed.

The always-valid fallback is `NullSoftPrompt` (no conditioning).

## 4. Data

### 4.1 Matched positives and negatives (from the same proteomes)

- **Source:** for each of the ~740 NCBI taxonomy groups that CARD genes come from (see
  `src/data/refseq_representative.py`), fetch the RefSeq reference/representative assembly's
  `protein.faa` and GFF3 (NCBI Datasets v2 API). If a strain-level taxid has no assembly, fall
  back to the species level.
- **Positives (binary task):** proteome proteins that hit CARD at ≥95% identity and ≥80%
  coverage (blastp). They come from the same proteomes as the negatives, so both classes share
  annotation pipeline, start-codon calls and coordinates. CARD's curated FASTA is **not** the
  positive source for the binary task.
- **Negatives:**
  - Drop anything with a CARD hit (all model types, including variant, overexpression and
    knockout).
  - Drop wild-type copies of CARD variant-model genes (gyrA, parC, rpoB, …).
  - Drop products whose names match resistance keywords.
  - Optionally drop anything AMRFinderPlus flags.
- **Tiers:**
  - **Easy:** no CARD hit at e ≤ 1e-3.
  - **Hard:** best CARD hit with 25–40% identity, ≥50% coverage, e ≤ 1e-5 (distant homologs).
- **Shortcut controls:**
  - length-matched (per length decile, using the same truncation rule as ESM-2)
  - taxonomy-matched (in proportion to each group's positive count)
  - 90%-identity deduplication (MMseqs2)
- **Diagnostics before any training:** check how well these alone separate the classes:
  - length + amino-acid composition (logistic regression)
  - the k-mer nearest-neighbour baseline

  If the easy tier is trivially separable, report that, and lean on the hard tier.

### 4.2 Outputs

`data/processed/negatives/`:
- `negatives.fasta`
- `negatives_metadata.tsv`
- `binary_positives_metadata.tsv`
- `negatives_summary.json`, with:
  - proteome coverage
  - how many proteins each filter removed
  - tier sizes
  - the length-matching KS statistic

Entry point: `scripts/build_negatives.py --config configs/negatives.yaml`.

## 5. Evaluation

1. **Cross-validation:** 5-fold `StratifiedGroupKFold`, seed 42, over train+val. The existing
   10% test split stays untouched as a final holdout. Two schemes:
   - **ARO-grouped:** matches the current split.
   - **Homology-partitioned:** groups are MMseqs2 clusters at 50% identity / 80% coverage. This
     is a stress test of generalization.
2. **Held-out whole genomes at real prevalence.** About 1% of a genome's ~4–6k proteins are
   AMR-like, so balanced accuracy is misleading. Report:
   - PR-AUC
   - precision among the top-k proteins flagged per genome
   - false positives per genome
   - calibration (reliability curve, ECE)
3. **Comparison against alignment-based tools:** RGI and AMRFinderPlus on the same genomes. The
   scientific claim is detecting **divergent** AMR-like proteins that alignment misses, so the
   hard tier and the homology-partitioned split carry most of the weight.
4. **Stage 2:** property-head accuracy on true positives. Also report end-to-end (detect then
   characterize) accuracy, so stage-1 errors are counted.
5. **Overfitting checks:** per-fold train−val gap, the epoch of the best val loss, and the
   paired difference between conditioned and null-control runs.

## 6. Injection mode for V3

- **Internal ("soft prompt", early fusion):** the context token is prepended inside the frozen
  ESM-2 encoder. This tests the paper's core contribution. It needs one ESM-2 pass for each
  conditioning value.
- **External ("late fusion"):** the context embedding is concatenated after mean-pooling. ESM-2
  embeddings can be cached once per protein and reused across both stages, so this is much
  cheaper and is **recommended for the tool**.

Both modes are ablated in the paper. Known audit items (see the main plan, Part 0):
- `transformers` must be pinned. V2 Runs 1 and 2 ran internal on `sjsu` (5.12.1) and external
  on `spark-833c` (5.13.0).
- Compare the prompt token's norm with ESM-2's word-embedding norm.
- Count sequences truncated at ~1022 aa.

## 7. Interface handoff contract (for the interface student)

Proposed inference entry point (built in this repo after the detector is validated):

```
predict(proteins: FASTA, context: optional per-protein {replicon_type, ta_proximity, mge_neighbor})
  -> per protein:
     {
       p_amr: float,                 # calibrated probability, stage 1
       is_amr_like: bool,            # at the documented threshold
       properties: {                 # stage 2, only when is_amr_like
         amr_gene_family: {label, confidence},
         drug_class:      [{label, confidence}, ...],
         resistance_mechanism: {label, confidence}
       },
       nearest_card_hit: {aro_accession, name, pident, coverage} | null
     }
```

Gene calling (Prodigal) and context extraction run before `predict` when the input is a raw
genome. The interface should present outputs as **"AMR-like"** with their confidence, never as a
clinical determination.

## 8. Build order

1. Docs (this file + CLAUDE.md) and the injection-mode audit.
2. CV folds + per-fold baselines (CPU). This also answers the open Run 2 leakage question.
3. ESM-2 embedding cache for external mode + parity gate. Cached results must reproduce
   `p4kl9z5r` within seed noise.
4. External-mode CV for V2 Runs 1–3, conditioned and null-control.
5. Proteome fetch → negative sampling → matched positives → genome-context features →
   diagnostics.
6. CV report. **After confirmation:** train the binary detector, then the stage-2 heads with
   context conditioning, then the genome-level benchmark against RGI/AMRFinderPlus.

## 9. Open decisions

- [ ] Andreopoulos confirms the V3 rescope.
- [ ] Bill: binary task shape, which context features go into the soft prompt, training negative
      ratio.
- [ ] Internal-mode CV scope (decide after the external-mode results).
- [ ] Held-out genome set for the benchmark (which organisms, and how many).
