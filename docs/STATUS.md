# AMR Soft Prompting — Project Status
_Last updated: 2026-08-04 17:05_

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
**V2 TA-proximity data layer ~35%** (TADB parsed; RefSeq fetch scope
resolved down to a concrete 738-accession list) — the RefSeq fetch itself,
BLAST coordinate mapping, distance binning, and the categorical embedding
are all still ahead. **V2 single-head restructuring 0%** — not started.

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
- Both TADB files and all CARD raw files are present on this server
  (`spark-833c`) at `data/raw/` (gitignored, as required).
- Full smoke suite: 152/152 passing outside `test_evaluate.py`. Two
  `test_train.py` failures seen in one combined run turned out to be GPU
  memory contention (see Blockers), not a real regression — they pass
  cleanly in isolation.

## What's In Progress

- Nothing actively running. `refseq_representative.py` (this session) is
  written and tested but not yet committed. Next actionable step is the
  RefSeq fetch itself, now that the 738-accession fetch list is resolved.

## What's Not Started

1. **RefSeq fetch + BLAST coordinate mapping** (CLAUDE.md TA-Proximity
   Pipeline Step 1) — pull the 738 representative accessions from RefSeq,
   pinned to CARD's own recorded version (not the current live version, to
   avoid coordinate drift from reannotation — confirmed approach with
   Aidan), then BLAST each of the 6404 resolvable CARD proteins against its
   organism group's representative accession for real start/end
   coordinates. No RefSeq data or fetch code exists yet.
2. **Same-replicon bp distance computation** (Pipeline Step 3) and the
   **categorical distance-bin embedding** (Pipeline Step 4, `ta_proximity.py`
   — bin edges deliberately deferred until the real distance histogram is
   available, per CLAUDE.md).
3. **V2 single-head restructuring** — three separate training runs (drug
   class / mechanism / gene-family targets per CLAUDE.md's task table),
   independently ablated on injection mode = 6 total runs. Needs new
   `ClassifierHead`/`SoftPromptModule` wiring per task, six new configs, and
   updated loss functions (`BCEWithLogitsLoss` for Run 1's multi-label drug
   class target). Not started in code — currently blocked behind the
   TA-proximity data layer above, since Run 3 needs it as conditioning
   input.
4. `src/data/prodigal_runner.py` — still an empty stub, deferred (unchanged
   from last entry).

## Open Questions / Blockers

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
- **Version-pinning decision recorded:** when the RefSeq fetch step is
  built, pin to CARD's exact recorded DNA Accession version rather than
  fetching the current live version, to keep BLAST coordinates
  self-consistent with the sequence CARD's protein was actually drawn from.
  Confirmed with Aidan this session, not yet implemented.
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
   lookup. 17/17 new tests passing. Not yet committed.
8. **Measured the accuracy cost of that dedup**: 83% (5316/6404) of CARD
   entries end up BLASTed against a substituted, non-own accession,
   concentrated almost entirely in a handful of large strain-diverse
   species (P. aeruginosa, A. baumannii, K. pneumoniae, E. coli) rather
   than spread evenly — flagged as an open question for Andreopoulos
   rather than silently accepted.
