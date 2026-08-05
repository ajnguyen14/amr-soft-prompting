# AMR Soft Prompting — Project Status
_Last updated: 2026-08-05 02:40_

## Current Version

**V2 — Single-Head Retargeting + TA-Proximity Conditioning.** CLAUDE.md's
roadmap and pipeline spec are fully up to date for this version, but the two
pieces of actual V2 code are at very different stages:

- **Single-head restructuring (3 separate runs, see CLAUDE.md's task table):
  not started in code.** `src/models/classifier.py` and
  `src/models/soft_prompt.py` still implement V1's fixed architecture
  (single `amr_gene_family` head; `resistance_mechanism` + `drug_class` as
  soft-prompt conditioning) — not the V2 matrix of three independent
  conditioning/target pairs. None of the six task-specific configs
  (`gpu_task{1,2,3}_..._{internal,external}.yaml`) exist yet; only the V1
  legacy `gpu_server_{internal,external}.yaml` pair is present.
- **TA-proximity conditioning pipeline (Run 3's input): in progress**, data
  layer only so far. See below.

## Completion

Rough estimate: **V1 unchanged from last entry (functionally complete)**.
**V2 TA-proximity data layer, Steps 1-3 code-complete and run end-to-end on
real data (~70%)** — Step 4 (categorical embedding) is blocked, not on
implementation effort but on a real finding: the pipeline currently
produces almost no usable signal (see "TA-proximity Steps 1-3 run on real
data" below and Blockers). **V2 single-head restructuring 0%** — not
started.

**Headline finding this session: Run 3's TA-proximity conditioning input,
as currently scoped, is very likely not viable.** Only 19 of 6404 CARD ARO
accessions (0.3%) resolve to a real same-replicon distance value. This
traces back to a hard ceiling established earlier this session
(card_tadb_matcher.py: only 28 of CARD's ~5,973 distinct replicons even
exist in TADB's replicon set at all, regardless of fetch/BLAST strategy) —
it is a real property of how little CARD's and TADB Type II's replicon
coverage overlap, not a bug. **Flag for Andreopoulos before Step 4 work
continues** — see Blockers for options.

**Scope change this session (confirmed with Aidan):** the CARD↔TADB
accession-matcher prefilter (145/6052 ARO accessions, 28 replicons) is no
longer used to *scope* the BLAST step — it undercounted true TA-proximity
signal by only attempting BLAST on replicons already known to carry TADB
loci, which would have forced everything else into `unknown` rather than
the more accurate `no_ta_locus` (see CLAUDE.md's vocabulary distinction).
The RefSeq fetch step now targets organism-deduplicated representative
accessions instead — 738 accessions covering all 6404 CARD entries with
resolvable taxonomy, instead of either the narrow 28-replicon prefilter or
a full 5,973-accession fetch. `card_tadb_matcher.py` isn't orphaned by this
— its accession-matching logic is still exactly what Step 3 (same-replicon
bp distance) needs once BLAST coordinates exist; it's just no longer the
thing that decides what to fetch in Step 1.

## What's Working

- Everything from V1 (unchanged): `card_parser.py`, `dataset.py`,
  `esm2_wrapper.py`, `soft_prompt.py`, `classifier.py`, `preprocess_card.py`,
  `run_training.py`, `load_config`, the label-leakage fix, both retrained V1
  ablation checkpoints (`i7o4eg5n` internal / `2rr2h1f9` external, 0.9087 /
  0.9054 val gene-family accuracy).
- **`src/data/tadb_parser.py` (this session).** Parses all four
  `type_II_{T,AT}_{exp,pre}.fas` files into `TADBLocus` records. Handles two
  data gotchas found while building it: minus-strand headers list
  coordinates largest-first (normalized to `start <= end` with strand
  tracked separately), and one row in `type_II_AT_pre.fas`
  (`AT240719`) has a coordinate in scientific notation (`2e+06`) instead of
  a plain integer. 14/14 tests passing, including against the full real
  dataset (403 exp toxin + 404 exp antitoxin + 169,035 pre toxin + 169,035
  pre antitoxin = 338,877 headers parsed with zero unhandled malformed
  rows).
- **`src/data/card_tadb_matcher.py` (this session, committed as `7015234`).**
  Version-strips CARD's `aro_index.tsv` `DNA Accession` field
  (`AL123456.3` → `AL123456`) and joins against TADB's already-unversioned
  replicon accessions — this is the CLAUDE.md TA-Proximity Pipeline Step 1
  prefilter that scopes the (not-yet-built) BLAST step to a bounded replicon
  set instead of a broad RefSeq subset. **Measured coverage: 145/6052
  (~2.4%) of CARD ARO accessions have a DNA Accession matching a TADB
  replicon.** Confirmed empirically that CARD's own `fmin`/`fmax` fields
  (in `card.json`) cannot substitute for BLAST — they're relative to CARD's
  own excised gene fragment (`fmax - fmin == len(sequence)` for all 6404
  numeric-coordinate entries, zero exceptions), not real replicon-level
  coordinates, so BLAST is still required to place matched genes within
  their replicon. `parse_aro_index` was promoted from private to public in
  `card_parser.py` (rename only, no behavior change) so the matcher could
  reuse it instead of re-parsing the TSV. 12/12 new tests passing, including
  against the real dataset.
- **`src/data/refseq_representative.py` (this session).** Resolves the
  RefSeq fetch scope for Step 1: joins CARD's `card.json` per-entry DNA
  accession to its NCBI taxonomy ID (6404 entries have both), groups by
  taxonomy ID into **740 organism groups**, and within each group picks the
  most-common CARD-recorded DNA accession as that organism's representative
  (ties broken lexicographically — fully deterministic, no external RefSeq
  lookup, so no reproducibility risk from NCBI's designations changing over
  time). Deduplicated fetch list: **738 accessions** (two groups converge on
  the same accession) — down from 5,973 unique CARD DNA accessions, a >8x
  reduction. 17/17 new tests passing, including against the real dataset.
  **Important caveat, measured not assumed:** the most-common rule only
  meaningfully preserves each ARO entry's own recorded accession for
  organisms with few distinct strain accessions (476/740 groups have only
  one accession to begin with, so no substitution happens there). For the
  handful of large, strain-diverse species — *P. aeruginosa* (1081 entries /
  1067 accessions), *A. baumannii* (737/729), *K. pneumoniae* (704/698),
  *E. coli* (562/554) — group size and candidate-accession count are nearly
  1:1, so "most common" barely helps: **5316/6404 (83%) of all ARO entries
  end up BLASTed against a substituted (non-own) accession**, concentrated
  in these few large groups rather than spread evenly. This is an inherent
  property of deduping by organism at all (no representative-selection rule
  avoids it for species this strain-diverse, short of not deduping), not a
  flaw specific to the most-common rule — but it's a real accuracy tradeoff
  against the fetch-count reduction and should be weighed before treating
  Run 3's TA-proximity signal as final. Not yet raised with Andreopoulos.
- **`src/data/refseq_fetch.py` + `scripts/fetch_refseq_representatives.py`
  (this session) — RefSeq fetch actually run.** Fetches nucleotide FASTA
  per representative accession via `Bio.Entrez` (`Entrez.email =
  aidan.j.nguyen@sjsu.edu`, no API key on file, so rate-limited client-side
  to NCBI's unauthenticated 3 req/sec ceiling), pinned to CARD's exact
  recorded accession version. Fetches are resumable — an existing
  `<accession>.fasta` on disk is never re-fetched, which matters because
  version-pinning makes a cached file provably not stale. **Run on
  `spark-833c` (CPU server offline this session) since this step is
  non-GPU, network-bound work — fine per the escalation rules either way.**
  First pass: 735/738 (99.6%) succeeded in ~12 min; 3 large genomes
  (`AM412317.1`, `CP000494.1`, `CP020412.2`) hit transient
  `IncompleteRead` truncation. Re-running the same script (resumable by
  design) fetched the remaining 3 in seconds — **738/738 (100%) now on
  disk** at `data/raw/refseq/` (625MB, gitignored). 9/9 new tests passing
  (mocked `Entrez.efetch`, no real network calls in the test suite).
  Config: `configs/ta_proximity_refseq.yaml` (new file, not a training
  config so doesn't touch the internal/external parity rule).
- **BLAST+ 2.17.0 installed on `spark-833c` (this session), no root
  required.** Not on PATH, not in the conda env, and `sudo apt install` was
  unavailable (no passwordless sudo). Installed via NCBI's precompiled
  tarball directly to `~/tools/ncbi-blast-2.17.0+/bin` instead of
  `conda install` (CLAUDE.md: pip only for new packages — moot anyway since
  BLAST+ isn't a Python package). **Architecture trap worth flagging for
  other sessions:** `spark-833c` is aarch64 (DGX Spark's Grace CPU, not the
  x86_64 typical of most servers) — the standard `x64-linux` BLAST+ tarball
  silently fails with `Exec format error` here; the `aarch64-linux` tarball
  is required. The RTX 3090 boxes are presumably x86_64 and would need the
  standard tarball instead.
- **`src/data/blast_runner.py` (this session).** Wraps `makeblastdb`/
  `tblastn` via subprocess (no first-class Python BLAST-running binding
  exists; `Bio.Blast` only parses output). Batches all queries for one
  representative accession into a single `tblastn` call (738 calls, not
  6404). Normalizes `tblastn`'s `sstart`/`send` to `start <= end` with
  strand tracked separately, matching `TADBLocus`'s convention. 8/8 tests
  passing against real (not mocked) BLAST+ binaries, using a real CARD gene
  (CblA-1) embedded in a synthetic genome with known coordinates — this
  also confirmed a real BLAST behavior worth having verified rather than
  assumed: a protein query has no stop-codon symbol, so the aligned subject
  range ends 3bp short of the full CDS.
- **`scripts/run_blast_coordinate_mapping.py` — full BLAST run, real data.**
  Groups CARD's 6052 protein sequences (from the training FASTA; 352 of the
  6404 taxonomy-resolved ARO accessions aren't in it and are skipped) by
  representative accession and BLASTs each group against its fetched
  genome. **Result: 1952/6052 (32.3%) BLAST-mapped.** Confirms and
  quantifies the accession-substitution risk flagged when
  `refseq_representative.py` was built: most of the 83% substituted
  entries simply have no homologous hit in their substituted genome at all
  (an AMR gene present in one strain is frequently just absent from a
  different strain's genome, not merely at a different coordinate).
- **`src/data/ta_proximity.py` (this session).** Step 3: classifies every
  ARO accession into `distance` (real same-replicon bp value to the
  nearest TADB locus), `no_ta_locus` (BLAST-mapped fine, no TA locus on
  that replicon), or `unknown` (BLAST failed). Version-strips BLAST hit
  replicons before comparing against TADB's unversioned ones; never
  compares across different replicons even from the same organism, per
  CLAUDE.md. Does **not** build Step 4's embedding vocabulary — bin edges
  are still explicitly deferred pending a usable histogram (see below, this
  is now the actual blocker, not just a sequencing choice). 9/9 tests
  passing. `scripts/run_ta_proximity.py` ran it against the real BLAST hits
  and full TADB loci set:

  ```
  6404 ARO accessions total
    19   distance    (0.3%)
  1933   no_ta_locus (30.2%)
  4452   unknown     (69.5%)
  ```

  **This is very likely not enough signal for a learned embedding** — see
  Blockers for the ceiling analysis and options.
- **Cross-checked against a second, much rosier run from a different
  server** (user-reported: 2397/6052 `distance`, 39.6%) and concluded it's
  very likely invalid, not a better result to reconcile toward. Reasoning:
  `card_tadb_matcher.py` already established, independent of any BLAST or
  dedup strategy, that only 28 of CARD's ~5,973 distinct replicons exist in
  TADB's replicon set *at all* — a hard ceiling of ~145 ARO accessions on
  how many could ever land near a TA locus under a correct same-replicon
  comparison. 2397 is 16x above that ceiling, which is very hard to explain
  except by a same-replicon check not actually being enforced (e.g.
  matching by organism/species instead of exact replicon accession —
  exactly the failure mode CLAUDE.md's pipeline spec explicitly warns
  against). **Not yet confirmed against that session's actual code** — the
  other session's owner couldn't check at the time this was raised.
- **Investigated whether expanding TADB scope beyond Type II could raise
  the ceiling (in progress, not yet acted on).** TADB 3.0 has Types I, III–
  VIII in addition to Type II, all downloadable in the same FASTA format
  (checked live against the TADB 3.0 download page + counted real records
  via curl, not from page text, since the site itself doesn't publish
  counts). Real per-file record counts on this session:

  | Type | exp (T+AT) | pre (T+AT) |
  |---|---|---|
  | I   | 200 | 54,677 |
  | III | 16  | 257 |
  | IV  | 27  | 33,370 |
  | V   | 2   | 5,572 |
  | VI  | 2   | 4 |
  | VII | 6   | 776 |
  | VIII| 8   | 13,956 |

  Combined, these would add ~108,900 loci on top of Type II's existing
  338,877 (a ~32% increase in raw loci) — Type I and Type IV are the
  largest additions. Whether this would meaningfully raise the 28-replicon
  overlap ceiling with CARD's specific replicon set is **not yet checked**
  — that requires actually running the accession-intersection logic against
  the new replicons, not just counting records. Header format for the new
  types (esp. Type I/VIII's RNA-based antitoxin entries) also not yet
  confirmed compatible with `tadb_parser.py`'s existing regex.
- Both TADB files and all CARD raw files are present on this server
  (`spark-833c`) at `data/raw/` (gitignored, as required).
- Full smoke suite: 187/187 passing outside `test_evaluate.py`/
  `test_train.py`. Earlier `test_train.py` failures traced to GPU memory
  contention (see Blockers), not a real regression.

## What's In Progress

- Nothing actively running. Steps 1-3 of the TA-proximity pipeline are
  code-complete and have been run on the full real dataset. Next decision
  point is whether/how to expand TADB scope (Type I/III-VIII) or otherwise
  address the sparse-signal finding — not yet resolved, see Blockers.

## What's Not Started

1. **Step 4: categorical distance-bin embedding** — blocked, not merely
   sequenced after Steps 1-3. With only 19 real `distance` values, bin
   edges "set from the actual histogram" (per CLAUDE.md) would be
   statistically meaningless. Needs a scoping decision first (see
   Blockers) before this is worth building.
2. **V2 single-head restructuring** — three separate training runs (drug
   class / mechanism / gene-family targets per CLAUDE.md's task table),
   independently ablated on injection mode = 6 total runs. Needs new
   `ClassifierHead`/`SoftPromptModule` wiring per task, six new configs, and
   updated loss functions (`BCEWithLogitsLoss` for Run 1's multi-label drug
   class target). Not started in code — currently blocked behind the
   TA-proximity data layer above, since Run 3 needs it as conditioning
   input.
3. `src/data/prodigal_runner.py` — still an empty stub, deferred (unchanged
   from last entry).

## Open Questions / Blockers

- **TOP BLOCKER: Run 3's TA-proximity signal is very likely too sparse to
  use, as currently scoped.** Only 19/6404 (0.3%) ARO accessions get a real
  distance value; the rest are `no_ta_locus` (30.2%) or `unknown` (69.5%).
  This traces to a hard, fetch/BLAST-strategy-independent ceiling: only 28
  of CARD's ~5,973 distinct replicons exist in TADB Type II's replicon set
  at all (card_tadb_matcher.py, established earlier this session), capping
  real matches at ~145 ARO accessions even in the best case. **Not a bug —
  a property of how little the two databases' replicon coverage overlaps.**
  A second run from a different server reported a much higher number
  (2397/6052, 39.6%) but is very likely invalid: it's 16x above the
  ceiling above, most plausibly because same-replicon-accession equality
  isn't actually being enforced there (CLAUDE.md explicitly warns against
  exactly this — comparing across replicons even from the same organism).
  Not yet confirmed against that session's code. **Options, not yet decided
  with Andreopoulos:**
  1. Collapse the conditioning input to a coarse 3-way categorical
     (`unknown`/`no_ta_locus`/`near_ta_locus`) instead of distance bins —
     usable even with ~145 positive examples, much less informative than
     originally intended.
  2. Expand TADB scope beyond Type II (Types I/III-VIII, ~108,900
     additional loci available — see What's Working) to raise the ceiling.
     Not yet checked whether this actually helps (needs the real
     accession-intersection re-run, not just more raw loci).
  3. Reconsider whether TA-proximity is viable as Run 3's conditioning
     input at all, given this ceiling.
- **Push access from this server was intermittently broken, now looks
  resolved.** Earlier in this session, `git push origin v2-ta-proximity`
  failed with `git@github.com: Permission denied (publickey)` — this server
  (`spark-833c`) has an SSH keypair at `~/.ssh/id_ed25519` that didn't
  appear authorized on GitHub. `origin/v2-ta-proximity` was later found
  in sync with `75af37d` without a push being run again in this session, so
  whatever was blocking it seems to have been fixed out-of-band. Not
  re-verified since — the latest commit (`7015234`, the accession matcher)
  is queued locally, not yet pushed. **Context:** this branch's earlier work
  (BLAST runner, TADB parser, TA-proximity module, and their preprocess
  scripts) was lost when a different, currently-unreachable server went
  down without that work ever being pushed — everything on this branch as
  of this session is being rebuilt from scratch specifically to avoid
  repeating that. Keep commits small and push often until that's confirmed
  reliable again.
- **GPU memory contention on `spark-833c` is worse than CLAUDE.md's existing
  note suggests.** `nvidia-smi` shows three unrelated `ollama` processes
  holding ~32GB combined. This caused a CUDA OOM in `test_evaluate.py` and,
  in one combined full-suite run, two `test_train.py` failures that could
  not be reproduced running that file alone. Not a code regression, but
  worth being aware any GPU-touching test/run on this host right now is
  competing with those processes.
- **Version-pinning decision: implemented.** `refseq_fetch.py` fetches
  CARD's exact recorded accession version (not the current live version),
  keeping BLAST coordinates self-consistent with the sequence CARD's
  protein was actually drawn from. No longer open.
- **Organism-dedup substitution rate (83%) not yet reviewed with
  Andreopoulos.** See `refseq_representative.py` entry above under What's
  Working — for the largest, most strain-diverse species in CARD, "most
  common accession per organism" barely reduces how often an ARO entry is
  BLASTed against a different strain's genome than its own. Before treating
  Run 3's TA-proximity signal as trustworthy for those species, worth a
  decision on whether this is acceptable or needs a different approach
  (e.g. not deduping the very largest groups, at the cost of more fetches).
- `docs/reviews/2026-07-05-classifier-loss-review.md` — remaining deferred
  items are low-severity doc/style nitpicks; unchanged.

## Recent Changes

1. **Rebuilt `src/data/tadb_parser.py` from scratch** on this server after
   the original (unpushed) implementation was lost with the server it was
   written on. Committed as `75af37d`.
2. **Built `src/data/card_tadb_matcher.py`**, the CARD↔TADB accession
   prefilter — first concrete coverage number for the TA-proximity pipeline
   (145/6052 CARD entries, ~2.4%). Committed as `7015234`.
3. **Confirmed CARD's `fmin`/`fmax` can't replace BLAST** for genomic
   coordinates — they're fragment-relative, not replicon-relative, checked
   against all 6404 numeric-coordinate `card.json` entries with zero
   exceptions.
4. **Promoted `card_parser._parse_aro_index` to public `parse_aro_index`**
   (rename only) so the new matcher could reuse CARD's existing TSV-parsing
   logic instead of duplicating it.
5. Checked out `v2-ta-proximity` on this server (`spark-833c`) to resume V2
   work; confirmed via `git log`/`git diff` that the branch's only prior
   remote commit was the unrelated batch-size fix (`9bd27eb`) — none of the
   lost BLAST/TADB/TA-proximity work had ever reached origin.
6. **Rescoped the RefSeq fetch step away from the accession-matcher
   prefilter, per Aidan.** The 28-replicon prefilter would have forced
   ~97.6% of CARD entries into `unknown` regardless of whether they
   actually have a nearby TA locus, conflating "BLAST not attempted" with
   the real `no_ta_locus` signal CLAUDE.md's vocabulary distinguishes.
7. **Built `src/data/refseq_representative.py`**, deduping CARD's 5,973
   distinct DNA accessions down to 740 organism (NCBI taxonomy ID) groups
   and selecting one most-common representative accession per group —
   738 accessions to fetch, deterministic and free of any external RefSeq
   lookup. 17/17 new tests passing. Committed as `9b59c49`.
8. **Measured the accuracy cost of that dedup**: 83% (5316/6404) of CARD
   entries end up BLASTed against a substituted, non-own accession,
   concentrated almost entirely in a handful of large strain-diverse
   species (P. aeruginosa, A. baumannii, K. pneumoniae, E. coli) rather
   than spread evenly — flagged as an open question for Andreopoulos
   rather than silently accepted.
9. **Built and ran the actual RefSeq fetch** (`src/data/refseq_fetch.py`,
   `scripts/fetch_refseq_representatives.py`, `configs/ta_proximity_refseq.yaml`)
   on `spark-833c` (CPU server offline this session). 3 large genomes hit
   transient `IncompleteRead` truncation on the first pass (735/738); the
   fetcher's resume-by-default behavior (skip files already on disk, safe
   because fetches are version-pinned) meant re-running the same command
   picked up just the 3 stragglers in seconds. **738/738 (100%) representative
   accessions now on disk** at `data/raw/refseq/` (625MB, gitignored).
   9/9 new tests passing (mocked `Entrez.efetch`, no real network calls in
   the test suite itself).
10. **Installed BLAST+ 2.17.0 on `spark-833c`** without root (NCBI's
    precompiled aarch64 tarball — the standard x64 build fails silently
    with `Exec format error` on this ARM64 host). Built and tested
    `src/data/blast_runner.py` (tblastn wrapper, 8/8 tests against real
    BLAST binaries). Committed as `eaa4469`.
11. **Built and ran `scripts/run_blast_coordinate_mapping.py` against the
    real dataset**: 1952/6052 (32.3%) CARD proteins BLAST-mapped onto their
    organism group's representative genome. Committed as `38c5b5d`.
12. **Built `src/data/ta_proximity.py` (Step 3) and ran it**: only
    19/6404 (0.3%) ARO accessions get a real same-replicon TA-locus
    distance; 1933 (30.2%) `no_ta_locus`; 4452 (69.5%) `unknown`. 9/9 tests
    passing. Committed as `350ea1e`.
13. **Investigated a conflicting, much higher result (2397/6052, 39.6%)
    reported from a different server session.** Concluded it's very likely
    invalid — 16x above a hard ceiling (~145 ARO accessions) established
    earlier this session via `card_tadb_matcher.py`, independent of fetch
    or BLAST strategy. Most plausible cause: the other pipeline isn't
    enforcing same-replicon-accession equality before computing a distance.
    Not yet confirmed against that session's actual code.
14. **Checked TADB 3.0's download page for data beyond Type II.** Types
    I/III-VIII exist (~108,900 additional loci beyond Type II's 338,877,
    counted directly via curl, not page text). Not yet determined whether
    this would raise the 28-replicon ceiling — flagged as one of three
    options for resolving the sparse-signal blocker, pending a decision
    with Andreopoulos rather than unilaterally expanded.
15. **Updated this file with the full session's findings** — the
    sparse-signal blocker is now the top item under Open
    Questions/Blockers, not buried under routine progress notes.
