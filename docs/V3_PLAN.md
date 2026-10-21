# V3 Plan — Binary AMR Detection + Feature Analysis

> **Draft for review by Professor Andreopoulos — revised 2026-10-19.**
> The scope was confirmed on 2026-10-09 (§9). Open decisions are listed in §9, and gaps to
> close before publication are listed in §10.

## 1. Goal

**Main task: binary prediction of AMR-like vs. non-AMR proteins.** Given a protein (and the
genome it came from), the model outputs a calibrated probability that it is an AMR gene.

**Second task: feature analysis on the trained detector.** After the detector works, we measure
how each biological property (replicon type, taxonomy, KEGG function, nearness to marker genes)
changes the prediction, and how the properties interact (§5b). This is **not** a correlation
study between features. Every feature claim is tied back to its effect on the binary prediction.

**Characterization (stage 2):** proteins flagged as AMR-like also get gene family, drug class
and mechanism predictions. Students see these, but they are not the main result.

Andreopoulos wants the project to become a tool that helps students learn about AMR *and* about
machine learning. The model will be put into a UI-based educational tool (§7).

Scope changes from the earlier V3:
- **KEGG returns to V3**, as a masked input feature (§3). It is not a RAG source.
- **RAG moves to V4 / unscheduled.**

### Ownership

The educational interface is owned by Professor Andreopoulos's team. Ownership of the data,
modelling and analysis work is still to be decided.

The **bioRxiv paper covers the model, feature analysis and benchmark.** The interface (§7) does
not block the preprint.

## 2. Pipeline

```
Genome (nucleotide FASTA)            ← tool input; benchmark uses RefSeq assemblies directly
      │
  [Gene calling: Prodigal]           ← src/data/prodigal_runner.py (stub → back in scope)
      │
Proteins + coordinates + features (§3)
      │
  [ESM-2 frozen] + [soft prompt ← one token per feature]
      │
  Stage 1: binary AMR-like detector        → p_amr per protein (calibrated)   ← MAIN RESULT
      │  (proteins above threshold)
  Stage 2: property heads                   → gene family / drug class / mechanism
      │
Flagged AMR-like proteins, each with properties, confidence and nearest CARD hit
```

## 3. Features (soft-prompt conditioning)

V2 Runs 1 and 2 conditioned on the **true** `amr_gene_family`. A protein from a student's genome
has no such label. So in V3, every feature must be computable for **any** protein in an
annotated or Prodigal-called genome:

| Feature | Values | Source / how computed | Leakage notes |
|---|---|---|---|
| `replicon_type` | chromosome / plasmid / unknown | RefSeq assembly report + GFF; for raw genomes, contig header or a plasmid classifier | Real biology: AMR genes are enriched on plasmids |
| `taxonomy` | coarse rank (default: phylum; order as an alternative) + `other` | NCBI taxid → lineage (NCBI Datasets / taxdump) | Negatives are taxonomy-matched (§4.1), so this cannot separate classes by itself. It can only help through interaction with sequence. Still gated |
| `kegg_category` | KEGG top-level pathway/BRITE category + `unassigned` | `kofam_scan` (local KofamKOALA HMMs) → KO → category | **Mask** before use: BRITE ko01504 (antimicrobial resistance genes), maps ko01501/ko01502/ko01503 (resistance pathways), and any KO whose definition matches resistance keywords. A masked KO becomes `unassigned`. The gate is mandatory |
| `ta_proximity` | `distance` / `no_ta_locus` / `unknown` | TADB 3.0 header coordinates on the same replicon (reuse `src/data/tadb_parser.py`, `ta_proximity.py`), using native proteome coordinates with no BLAST mapping | Same 3-way vocabulary as V2 Run 3. Raw student genomes need a TA detection path (TA HMMs / TAfinder), see §9 |
| `mge_proximity` | `near` / `not_near` / `unknown` | Transposase, integrase, recombinase or IS element within N kb (N set in config), from GFF product names or Pfam HMMs | Exclude resistance-keyword products from the "marker" set |
| `integron_proximity` | `near` / `not_near` / `unknown` | `intI` integrase within N kb | Same as above |

**Explicitly excluded:** nearness to *other AMR genes*. Computing it needs the labels of
neighbouring proteins, so it leaks the target, and at inference it would depend on the model's
own predictions.

**Multi-feature conditioning:** one soft-prompt token per feature, each an `nn.Embedding`.
This extends the `SingleFieldSoftPrompt` pattern in `src/models/soft_prompt.py` into a
multi-token module. It is an **architectural change to the soft prompt module**, so its design
is written up and flagged for review before it is implemented. `NullSoftPrompt` stays as the
always-valid control.

**Leakage gate (required before any feature is used):** score each feature alone (the
`conditioning_lookup` baseline in `src/eval/baselines.py`) on AMR vs. non-AMR.
- If it separates the classes because of real biology, it may be used, but its score becomes a
  baseline the model must beat.
- If it separates them only because of how the data was built, drop it.
- A "non-AMR" token, or any value that only negatives can take, is never allowed.

## 4. Data

### 4.1 Matched positives and negatives (from the same proteomes)

- **Source:** for each of the ~740 NCBI taxonomy groups that CARD genes come from (see
  `src/data/refseq_representative.py`), fetch the RefSeq reference/representative assembly's
  `protein.faa`, GFF3 and assembly report (NCBI Datasets v2 API, reusing `src/data/refseq_fetch.py`).
  If a strain-level taxid has no assembly, fall back to the species level.
- **Positives:** proteome proteins that hit CARD at ≥95% identity and ≥80% coverage (blastp, via
  `src/data/blast_runner.py`). They come from the same proteomes as the negatives, so both classes
  share annotation pipeline, start-codon calls and coordinates. CARD's curated FASTA is **not** the
  positive source for the binary task. It stays the source for stage-2 property labels, joined by
  the matched ARO.
- **Negatives:**
  - Drop anything with a CARD hit (all model types, including variant, overexpression and
    knockout).
  - Drop wild-type copies of CARD variant-model genes (gyrA, parC, rpoB, …).
  - Drop products whose names match resistance keywords.
  - Optionally drop anything AMRFinderPlus flags.
- **Tiers:**
  - **Easy:** no CARD hit at e ≤ 1e-3.
  - **Hard:** best CARD hit with 25–40% identity, ≥50% coverage, e ≤ 1e-5 (distant homologs).
  - Proteins between the tiers (e.g. 40–95% identity) are neither positive nor negative. They are
    excluded from training and reported as a count.
- **Shortcut controls:**
  - length-matched (per length decile, using the same truncation rule as ESM-2)
  - taxonomy-matched (in proportion to each group's positive count)
  - 90%-identity deduplication (MMseqs2)
- **Training ratio:** 1 positive : 10 negatives by default (easy + hard, hard tier
  over-represented at 50% of negatives). Configurable in `configs/negatives.yaml`. Evaluation is
  always also done at real genome prevalence (§5).

### 4.2 Dataset spec (fill in counts after step 3 of §8)

| Quantity | Count |
|---|---|
| Taxonomy groups queried / with an assembly | ~740 / TBD |
| Proteins scanned | TBD |
| Positives (≥95% id, ≥80% cov) / unique AROs covered | TBD / TBD |
| Easy negatives (after filters, dedup, matching) | TBD |
| Hard negatives | TBD |
| Excluded grey-zone proteins | TBD |
| Feature coverage (% non-`unknown`) per feature | TBD |

### 4.3 Outputs

`data/processed/v3/`:
- `positives.fasta`, `negatives.fasta`
- `binary_dataset.tsv`: one row per protein, with columns:
  `protein_id, assembly_accession, replicon_accession, start, end, strand, taxid, phylum, order,
  label (amr / non_amr), tier (positive / easy / hard), card_aro, card_pident, card_coverage,
  card_evalue, kegg_ko, kegg_category, replicon_type, ta_proximity, mge_proximity,
  integron_proximity, length, split_fold`
- `dataset_summary.json`, with:
  - proteome coverage
  - how many proteins each filter removed
  - tier sizes
  - the length-matching KS statistic
  - per-feature coverage and leakage-gate scores

Entry points:
- `scripts/build_binary_dataset.py --config configs/negatives.yaml` (positives, negatives, tiers)
- `scripts/extract_features.py --config configs/negatives.yaml` (the §3 columns)

### 4.4 Diagnostics before any training

Check how well these alone separate AMR from non-AMR:
- length + amino-acid composition (logistic regression)
- the k-mer nearest-neighbour baseline
- each feature alone (the leakage gate)

If the easy tier is trivially separable, report that, and lean on the hard tier.

## 5. Evaluation

1. **Cross-validation:** 5-fold `StratifiedGroupKFold`, seed 42, over train+val. A 10% test
   split stays untouched as a final holdout. Two schemes:
   - **Assembly-grouped** (binary task): no proteome is split across folds. The V2 re-runs in
     §8 step 2 keep the ARO-grouped scheme that matches the current split.
   - **Homology-partitioned:** groups are MMseqs2 clusters at 50% identity / 80% coverage. This
     is a stress test of generalization.
2. **Held-out whole genomes at real prevalence.** About 1% of a genome's ~4–6k proteins are
   AMR-like, so balanced accuracy is misleading. Report:
   - PR-AUC (main metric)
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

### 5b. Feature analysis (the second task)

All of this is run on the trained binary detector. External mode makes it cheap, because
ESM-2 embeddings are cached once per protein.
- **Ablation:** null control → each feature alone → all features → all minus one. Report the
  change in PR-AUC (with fold-level paired differences) for each feature.
- **Counterfactual swaps:** for each protein, change one feature (e.g. chromosome → plasmid,
  `no_ta_locus` → `distance`) and record the shift in p_amr. This answers "how does one property
  affect the prediction?" Pairs of swaps measure interactions between two features.
- **Stratified performance:** PR-AUC and false positives by taxonomy group, KEGG category and
  replicon type, to show where the model works and where it does not.
- **Descriptive tables:** feature × label counts, as context for the effects above. These are
  never reported as findings on their own.

## 6. Injection mode for V3

- **Internal ("soft prompt", early fusion):** the feature tokens are prepended inside the frozen
  ESM-2 encoder. This tests the paper's core contribution. It needs one ESM-2 pass for each
  combination of feature values, which makes counterfactual analysis expensive.
- **External ("late fusion"):** the feature embeddings are concatenated after mean-pooling. ESM-2
  embeddings can be cached once per protein and reused across both stages and all counterfactuals.
  It is much cheaper and is **recommended for the tool and for §5b**.

Both modes are ablated in the paper for the binary task. Known audit items:
- `transformers` must be pinned. V2 Runs 1 and 2 ran internal on `sjsu` (5.12.1) and external
  on `spark-833c` (5.13.0).
- Compare the prompt token's norm with ESM-2's word-embedding norm.
- Count sequences truncated at ~1022 aa.

## 7. Educational interface

The trained model will be put into a UI-based educational tool for bioscience and data science students. Its design is outside this project scope.

## 8. Build order (all on `sjsu`)

CPU-only steps also run on `sjsu` (RTX 3090). 650M GPU steps use batch ≤ 24.

1. **Docs + audit:** this file, CLAUDE.md; pin `transformers`; injection-mode audit (§6).
2. **CV folds + per-fold baselines** for the existing V2 tasks. This also answers the open Run 2
   leakage question: both modes reached ~99.5%, which looks like the pre-fix leakage pattern, so
   we need to check whether `amr_gene_family` alone predicts `resistance_mechanism`.
3. **Dataset build**: proteome fetch (reuse `src/data/refseq_representative.py`,
   `src/data/refseq_fetch.py`, `scripts/fetch_refseq_representatives.py`) → CARD blastp
   (`src/data/blast_runner.py`) → positives, easy/hard negatives, filters, matching, dedup →
   `scripts/build_binary_dataset.py`. Fill in §4.2.
4. **Feature extraction**: replicon type, taxonomy, `kofam_scan` + KEGG mask,
   TA/MGE/integron proximity (reuse `ta_proximity.py`, `tadb_parser.py`) →
   `scripts/extract_features.py`. Run the leakage gate per feature.
5. **Shortcut diagnostics** (§4.4). Stop and report if the easy tier is trivially separable.
6. **ESM-2 embedding cache** (GPU, external mode) + parity gate: cached results must reproduce
   the V2 Run 3 external checkpoint (`p4kl9z5r`) within seed noise (tolerance: see §10, item 9).
7. **Binary detector**, external mode: null control → each feature alone → all features. Then
   internal mode for the comparison. Multi-token soft prompt design is reviewed before this step.
8. **Feature analysis** (§5b), then the stage-2 heads conditioned on features.
9. **Genome-level benchmark** vs RGI/AMRFinderPlus.

New tools: `kofam_scan` (+ KOfam HMM profiles), MMseqs2, RGI, AMRFinderPlus, and HMMER for the
MGE/TA HMMs. Python packages go in `requirements.txt` (pinned, pip). Non-pip binaries are listed
with versions and install commands in `docs/V3_TOOLS.md` when they are installed.

## 9. Open decisions

- [x] Andreopoulos confirms the V3 rescope (2026-10-09).
- [x] Task shape: binary AMR vs. non-AMR, with feature analysis on top (2026-10-09).
- [x] KEGG in V3 as a masked category feature (2026-10-09).
- [ ] Ownership of the data, modelling and analysis work.
- [ ] Taxonomy rank for the feature (phylum vs. order).
- [ ] MGE/integron window N (kb).
- [ ] Training negative ratio and the hard-tier share (default 1:10, 50% hard).
- [ ] TA detection for genomes outside RefSeq/TADB (TA HMMs vs. TAfinder).
- [ ] Internal-mode CV scope (decide after the external-mode results).
- [ ] Held-out genome set for the benchmark (which organisms, and how many).
- [ ] Label source for "divergent AMR-like" proteins: the hard tier is currently labelled
  non-AMR, which conflicts with the §5.3 claim (see §10, item 1).

## 10. Reproducibility concerns

> These gaps must be closed before results are publishable. They are ordered by priority. Most are
> about pinning versions and spelling out procedures, not redesign. Item 1 is about scientific
> validity and is also an open decision in §9.

1. **Hard-tier labels conflict with the main claim (validity).**
   - Proteins at 25–40% identity to CARD are labelled *non-AMR* (§4.1). But §5.3 claims the model
     finds *divergent* AMR-like proteins that alignment misses. A model that does this is scored as
     making false positives.
   - Positives are defined by a ≥95% CARD hit, so RGI (which uses CARD) recovers them almost by
     construction. That makes the comparison against RGI circular.
   - Fix: use a source of "divergent AMR" labels that doesn't come from CARD (e.g. validated
     distant homologs, or a CARD release held back in time). Otherwise, frame the hard tier as a
     specificity test.
2. **No data snapshot or version manifest.** None of these are recorded:
   - the CARD release used for V3 (V1 used v4.0.1)
   - the assembly `accession.version` list (NCBI's "representative" status changes over time)
   - the KOfam profile and ko_list release, and the KEGG BRITE/map release used for the mask
   - the RGI and AMRFinderPlus database versions

   Fix: commit a manifest with checksums, and deposit `data/processed/v3/` on Zenodo with a DOI.
3. **The proteome source doesn't match the reused code.** §4.1 says "RefSeq
   reference/representative assembly". But `src/data/refseq_representative.py` picks the most
   frequent CARD-recorded DNA accession per taxid, which can be a plasmid rather than an assembly.
   `docs/STATUS.md` reports an 83% substitution rate. Fix: state one selection rule, its
   tie-break, and how strain-to-species fallback is decided.
4. **Gaps in the dataset-building rules.**
   - **Uncovered protein classes:** these fall into no tier. Examples: a best hit at
     1e-5 < e ≤ 1e-3; 25–40% identity but <50% coverage; <25% identity with e ≤ 1e-3; ≥95%
     identity but <80% coverage. Each needs an explicit bucket.
   - **blastp settings are missing:** DB build, `max_target_seqs`, composition-based stats,
     whether coverage is on the query or the subject, and the positive-hit e-value.
   - **"Resistance keywords" has no list.** It is used four times; commit it as a file.
   - **Undefined filter:** there's no stated way to identify "wild-type copies of CARD
     variant-model genes".
   - **AMRFinderPlus filter:** make the optional filter a config flag that gets recorded.
   - **Matching order and seed:** dedup, length deciles, taxonomy proportions, then the 1:10 /
     50%-hard ratio. These can conflict, so state what happens when they can't all be met. Add
     MMseqs2 parameters (coverage mode, cluster mode).
5. **Feature extraction is underspecified.**
   - **Plasmid classifier:** not named.
   - **GFF replicon rule:** no rule for setting `replicon_type` from a GFF.
   - **kofam_scan:** no threshold mode, and no rule for a protein with several KOs or a KO in
     several categories.
   - **MGE markers:** no Pfam HMM list or version.
   - **Open §9 choices:** taxonomy rank, MGE/integron window N and the TA detection path must
     each be frozen in a config before results are final.
6. **No training config for the detector.**
   - **Not specified:** learning rate, epochs, early stopping, model-selection criterion,
     calibration method and the split it's fit on, the `is_amr_like` threshold rule, stage-2 head
     settings.
   - **Configs:** none are named (`configs/v3_*.yaml`), and the internal/external parity rule
     isn't restated for V3.
   - **Seeds:** only seed 42 is used. Publication needs several seeds or confidence intervals,
     plus a multiple-comparison correction across the ablation grid.
7. **The environment isn't pinned.**
   - **requirements.txt:** torch, fair-esm, transformers, biopython, numpy, pandas and pytest are
     unpinned. §6 only flags transformers.
   - **Lockfile:** there is no lockfile or container.
   - **GPU determinism:** nothing covers cuDNN flags or `torch.use_deterministic_algorithms`.
   - **Tool versions:** `docs/V3_TOOLS.md` must exist before publication.
8. **No single reproduce entry point.** CLAUDE.md asks for one runnable entry point per pipeline.
   V3 spreads across `fetch_refseq_representatives.py`, `build_binary_dataset.py`,
   `extract_features.py` and external binaries. Fix: add a top-level reproduce script or Makefile
   that runs them in order.
9. **Splits and benchmark aren't fixed up front.**
   - **Test holdout:** the grouping (by assembly or by homology cluster) isn't stated.
   - **Benchmark genomes:** the §9 decision must be made and frozen *before* training to avoid
     selection bias.
   - **Parity gate:** "within seed noise" of `p4kl9z5r` has no tolerance, and only one seed was
     run.
   - **RGI/AMRFinderPlus modes:** not stated (e.g. RGI strict/perfect vs. `--include_loose`).
**Priority:**
- Items 2, 6, 7 and 8 are routine fixes: manifests, configs, pins, a reproduce script.
- Items 3, 4, 5 and 9 need concrete rules before step 3 of §8.
- Item 1 needs a decision (§9) before the dataset is built.
