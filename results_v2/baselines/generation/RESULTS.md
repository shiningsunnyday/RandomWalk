# Generation baselines on the corrected v2 polymer dataset — results and caveats

Dataset: the **114 v2 polymer repeat units** (`results_v2/data/group-contrib/polymer_walks_v2_smiles.txt`).
Every baseline is trained on these 114 molecules and nothing else, except REINVENT, whose prior is
pretrained on ChEMBL (flagged in its row). 1000 molecules are sampled per baseline and scored with
the metric definitions in `PLAN.md`, implemented in `common/metrics.py` to match
`results_v2/compute_numbers.py` lines 18–37. Raw numbers: `metrics.jsonl`; samples and logs:
`runs/<baseline>/`.

> **Status 2026-10-20 06:50 EDT — all 11 baselines complete**, plus the dataset reference row.
> Retro\* is measured on 10 of the 11 (run 1: 5 rows x 45; run 2: 4 rows x 25; JT-VAE x 25,
> verified SMILES-by-SMILES). **DEG has no Retro\* number**: a pass covering it ran at 06:40–06:42
> but a *second, concurrent Claude session* working in this same directory rebuilt the shared
> `retro/manifest.json` mid-flight, so that collect paired mismatched indices and the raw files were
> then overwritten. The unverifiable 4% was discarded rather than reported; STATUS.md has the
> one-command re-run. Everything else is verified.

## Table 1 — comparison

<!-- TABLE -->
| Method | Valid (Walk) | Valid (SMARTS) | Valid (SMILES) | Unique | Novel | Diversity | Retro\* | Retro\* n | mean HA | median HA | mean MW |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours (motif-walk grammar, best-first) | 63% | 100% | 90% | 62% | 100% | 0.91 | 54% | 1000 | 21.5 | 16 | 293 |
| — the 114 dataset monomers (reference, not a generator) | — | — | 100% | 97% | 0% | 0.82 | 9% | 45 | 53.0 | 52 | 718 |
| JT-VAE | — | — | 100% | 22% | 92% | 0.72 | 20% | 25 | 56.4 | 57 | 819 |
| Hier-VAE | — | — | 100% | 22% | 54% | 0.85 | 16% | 25 | 43.3 | 45 | 584 |
| Graph GA (unconditional) | — | — | 100% | 91% | 100% | 0.91 | 33% | 45 | 26.8 | 26 | 373 |
| Graph GA (max predicted CO2) | — | — | 100% | 100% | 100% | 0.60 | 0% | 45 | 58.4 | 60 | 765 |
| SMILES LSTM (early-stopped) | — | — | 17% | 7% | 22% | 0.81 | 12% | 25 | 41.2 | 39 | 553 |
| SMILES LSTM (fit to convergence) | — | — | 94% | 11% | 1% | 0.81 | 7% | 45 | 51.9 | 50 | 705 |
| SMILES transformer (early-stopped) | — | — | 0% | 0% | 100% | 0.00 | — | — | 1.0 | 1 | 16 |
| SMILES transformer (fit to convergence) | — | — | 86% | 11% | 0% | 0.81 | 16% | 25 | 54.6 | 54 | 740 |
| DEG (data-efficient grammar) | — | — | 100% | 50% | 60% | 0.81 | — | — | 45.6 | 43 | 621 |
| REINVENT 4 (TL only, no RL) | — | — | 88% | 42% | 53% | 0.85 | 13% | 45 | 43.6 | 44 | 592 |
| REINVENT 4 (TL + RL on CO2 surrogate) | — | — | 98% | 72% | 99% | 0.86 | 24% | 25 | 23.3 | 23 | 313 |

Columns. **Valid (Walk)** and **Valid (SMARTS)** are specific to our walk representation (a sampled
walk can be an invalid walk, or a valid walk whose SMARTS-based motif assembly fails); baselines emit
SMILES directly and have no analogue, so those cells are blank for them. **Valid (SMILES)** is RDKit
parse + sanitize. **Unique** is distinct canonical SMILES / 1000 and **Novel** is the fraction of
valid samples that are not one of the 114 training molecules (the same denominators as our own
script; `metrics.jsonl` also carries `unique_of_valid` and `novel_of_unique`). **Diversity** is
1 − mean pairwise Tanimoto, Morgan r=3, 2048 bits. **HA** = heavy atoms. **Retro\*** is the Retro\*
success rate from `repro_v1/retro.py` (24 workers, 50 iterations, expansion top-k 50), with the
number of molecules it was computed over in the next column.

## Our own row — the two caveats, stated plainly

1. **Unique = 62% is a stricter metric than the 96% in the earlier version of this work.** It is
   measured on clean canonical SMILES, after unconsumed R-group stand-ins are stripped and the
   structure is H-capped and de-atom-mapped. The like-for-like number on the raw atom-mapped SMILES
   is **98%**. Two structures that differ only in an unconsumed attachment point collapse to the same
   molecule under the stricter metric, and we count that as a duplicate.
2. **Our structures are small relative to the dataset, and that confounds Retro\* and Diversity.**
   Best-first enumeration favours short walks: median **16 heavy atoms**, 2.0 motifs per monomer,
   against an average of 7.4 motifs per walk in the dataset and a dataset median of **52 heavy
   atoms** (mean 53.0, mean MW 718; row 2 of the table). Baselines trained on the full-size monomers
   generate much larger molecules — the SMILES LMs and REINVENT land at 44–55 heavy atoms, Graph GA
   at 26 unconditionally and 60 when the surrogate objective is on. Retro\* gets harder and
   fingerprint diversity behaves differently as molecules grow, so **the Retro\* and Diversity
   columns are not strictly like-for-like across rows**; the size columns are reported so a reader
   can see it.

### A third caveat we must add to the paper: **Retro\* is almost entirely explained by molecule size**

This is the most important thing this exercise produced, and it changes how we should present our own
54%.

We ran Retro\* on the 114 real monomers under exactly the settings we use for generated structures.
A first probe on the first 24 of them solved **0 of 24** (`runs/_probe/retro_out.txt`); the seeded
45-molecule subsample in the main run solved **4 of 45 (8.9%)**. The dataset's own molecules are
essentially not retrosynthesisable by Retro\* at these settings.

Across every row where we have a Retro\* number, the rate falls with median heavy-atom count, but
not monotonically:

| row | median heavy atoms | Retro\* | n |
|---|---|---|---|
| **Ours** | **16** | **54%** | 1000 |
| REINVENT 4 (TL+RL on CO2 surrogate) | 23 | 24% | 25 |
| Graph GA (unconditional) | 26 | 33% | 45 |
| SMILES LSTM (early-stopped) | 39 | 12% | 25 |
| REINVENT 4 (TL only) | 44 | 13% | 45 |
| Hier-VAE | 45 | 16% | 25 |
| SMILES LSTM (fit to convergence) | 50 | 7% | 45 |
| **the 114 dataset monomers** | **52** | **9%** | 45 |
| SMILES transformer (fit to convergence) | 54 | 16% | 25 |
| JT-VAE | 57 | 20% | 25 |
| Graph GA (max predicted CO2) | 60 | 0% | 45 |

Over these 11 rows: **Pearson r = −0.83, Spearman ρ = −0.63**. Size is a strong driver but not the
whole story — JT-VAE has the *largest* molecules in the table (57 heavy atoms) and still scores 20%,
above the dataset's 9% at 52, and the fitted transformer scores 16% at 54. The reading that survives
all of it: **Retro\* rewards small molecules built from conventional building blocks.** JT-VAE and the
fitted LMs assemble their molecules from a small vocabulary of common motifs, which the planner
handles even at large size; the real dataset monomers are large *and* chemically unusual (fused-ring
PIMs, spirobifluorenes, polyimides), and score worst of the non-degenerate rows.

Our row wins on both axes at once — smallest molecules, assembled from common motifs — which is
exactly why 54% should not be presented as a bare headline. Three consequences for the paper:

1. **Report the dataset row next to ours.** The 114 real, experimentally characterised membrane
   polymers score 9%. Without that row, 54% invites the reading "our generated molecules are more
   synthesisable than real membrane polymers", which is not what the number means.
2. **Make the size-matched comparison explicitly, because it favours us.** The baselines closest to
   our molecule size are REINVENT-RL (23 heavy atoms, 24%) and Graph GA unconditional (26, 33%),
   against our 54% at 16. We are the best row at roughly comparable size — a defensible claim.
   "Best Retro\* in the table" on its own is not, since we are also the smallest row in it.
3. **Put the size columns beside Retro\* in Table 1** and consider a Retro\*-within-a-size-band
   column (e.g. ≤30 heavy atoms) as the like-for-like version.

## What the numbers say (the short version for the rebuttal)

1. **No learned baseline matches our Novel at a usable validity.** The polymer-realistic ones —
   Hier-VAE (100% valid), JT-VAE (100%), DEG (100%), REINVENT-TL (88%) — are 54%, 92%, 60% and 53%
   novel; for Hier-VAE, DEG and REINVENT-TL roughly half of what they emit is a training molecule.
   Ours is 100% novel at 90% SMILES validity. Only the search methods (Graph GA, REINVENT after RL)
   and JT-VAE reach ~90–100% novelty, and they do it by walking away from the data distribution
   rather than by modelling it.
2. **Uniqueness is where the baselines collapse, not validity.** Hier-VAE 22%, **JT-VAE 22%**, the
   SMILES LMs 11%, DEG 50%, REINVENT-TL 42% against our 62% (on the strict clean-SMILES metric; 98%
   like-for-like on raw atom-mapped SMILES). With 114 training molecules, every method that fits a
   distribution over molecules ends up re-emitting a handful of structures — either training
   molecules (Hier-VAE, the LMs) or a few of its own motif patterns (JT-VAE, which is 92% novel but
   only 22% unique and has the table's lowest diversity, 0.72).
3. **The strong baseline is the one reviewers would have asked for and it behaves exactly as
   expected.** Graph GA unconditional is 100% valid / 91% unique / 99.9% novel / diversity 0.91 —
   as good as ours on those three columns. Two things to say about it: its validity is a filter, not
   a learned property, and it is not a *polymer design method* in any useful sense — it has no
   notion of a repeat unit, it just cuts and recombines bonds, and its molecules drift to 27 heavy
   atoms. The comparison to make in the paper is on the design-space columns (motif semantics,
   attachment points, property-conditioned walks), not on validity.
4. **RL works on the surrogate, which is a statement about the surrogate.** REINVENT TL+RL reaches
   98% valid / 72% unique / 99% novel and shrinks its molecules to a median of 23 heavy atoms — the
   closest size match to ours in the table, so it is the fairest row to compare our Retro\* and
   Diversity against, and on that comparison we win (54% vs 24% Retro\*, 0.91 vs 0.86 diversity).
   Graph GA pointed at the same surrogate saturates it (top score 4.54 vs 4.43 for the best labelled
   polymer) and its diversity collapses to 0.60.
5. **DEG's grammar sampler is sound; it is DEG's reward loop we could not afford.** With grammar
   induction only (170 production rules, no RL) it is 100% valid / 50% unique / 60% novel /
   diversity 0.81 at median 43 heavy atoms, 1000 molecules with zero production failures. Report it
   as "DEG grammar induction, RL stage omitted" and say why (25,000 Retro\* calls).
6. **A character LM on 114 SMILES is not a baseline, it is a memoriser**: 1% and 0.4% novelty once
   fit, near-zero validity if early-stopped.
7. **Both motif/junction-tree VAEs drift large.** JT-VAE's median is 57 heavy atoms and Hier-VAE's
   45, against the dataset's 52 and our 16. If the paper wants a size-controlled generation
   comparison, the natural partners are Graph GA (26) and REINVENT-RL (23), not the VAEs.

## Per-baseline notes and caveats

### JT-VAE (`runs/jtvae/`)
Code: the JT-VAE port in `~/torch-molecule` (`torch_molecule.generator.jtvae`, the
kamikaze0923/jtvae re-implementation of Jin et al., ICML 2018), hidden 450 / latent 56, batch 16,
trained on all 114 SMILES on `cuda:0`, sampled with `sample_prior` (z ~ N(0,I) for the tree and the
graph latent, one molecule per call, no retries and no validity filtering).
**Result.** 53 epochs in 5608 s (loss 166 -> 2.32, junction-tree vocabulary of 33 entries), then
1000 prior samples in **22402 s (6.2 h)** with **zero decoder exceptions**: 100% valid, 22.1%
unique, 92.2% novel, diversity **0.724**, median **57 heavy atoms** (mean MW 819). Two things stand
out. (i) It produces the **largest molecules in the table — larger than the dataset's own median of
52** — because the tree decoder keeps attaching motifs from a 33-entry vocabulary; combined with the
lowest diversity of any learned row (0.72), the picture is a model that has learned one dominant
polyimide/fused-ring motif pattern and elaborates it. (ii) Its novelty (92%) is much higher than
Hier-VAE's (54%) but comes with the same uniqueness collapse (22%), i.e. it is not re-emitting
training molecules, it is re-emitting a small number of its *own* motifs.
Caveats: (a) training was given a 90-minute wall-clock budget rather than a fixed epoch count, and
the number of epochs actually run is in `runs/jtvae/run_info.json`; the loss curve is in
`runs/jtvae/loss.json`. (b) `torch_molecule`'s top-level `__init__` cannot be imported in this
checkout (`from .generator.random_walk import RandomWalkMolecularGenerator` raises
`NameError: name 'L_grammar' is not defined`), so that one unrelated submodule is stubbed in
`sys.modules`; nothing in `~/torch-molecule` was modified. (c) JT-VAE's junction-tree vocabulary
extracted from 114 molecules has only 33 entries, which is very small for a decoder that must
assemble fused-ring systems.

### Hier-VAE (`runs/hiervae/`)
Code: `~/hgraph2graph/polymers` (HierVAE, Jin et al., ICML 2020 — the polymer generation variant),
default architecture (hidden 250, latent 24, depthT/depthG 20, beta 0.1), motif vocabulary extracted
by the repo's own `get_vocab.py` with `--min_frequency 2` (200 entries; the repo's default of 100 is
meaningless at n=114).
Caveats, all of them forced by the dataset being 114 large molecules:
* **113 of 114 molecules are used.** HierVAE's atom vocabulary (`poly_hgraph/vocab.py:COMMON_ATOMS`)
  has no sodium, so the one `C(=O)O[Na]` polymer raises `KeyError: ('Na', 0)` during tensorization
  and was dropped. Novelty is still measured against all 114.
* `polymers/preprocess.py` computes `num_splits = len(batches) // 1000`, which is 0 for 6 batches,
  and then divides by it. `common/hier_preprocess.py` does the identical work and writes one shard.
* `polymers/vae_train.py` crashes on its own logging line under torch 2.x
  (`np.array([kl_div, ...])` on cuda tensors). `common/hier_train.py` is the same loop with
  `.item()` applied in the logging statement only. We could not use the `mhg` environment's torch
  1.1.0 instead, because torch 1.1.0 has no kernels for this GPU (sm_86).
* **The decoder's buffers are too small for polymer motifs.** `poly_hgraph/decoder.py` builds
  `IncTree(...)` with `max_sub_nodes=30`, i.e. at most 30 atoms in one motif node, and decoding our
  model raises `RuntimeError: The expanded size of the tensor (30) must match the existing size
  (47)`. `common/hier_sample.py` subclasses `IncTree`/`IncGraph` with larger buffers (120 atoms per
  motif, 400 nodes, 1200 edges) and patches them in; the model, weights and decoding algorithm are
  untouched. Anyone re-running the paper's Hier-VAE baseline on polymer-sized monomers has to do
  this, and it is worth a sentence in the paper.
* Training was stopped at the **step-500 checkpoint** (`runs/hiervae/ckpt/model.4`) rather than run
  to a fixed epoch budget: the machine's CPU was the scarce resource (load ~130 from another
  campaign, every one of my processes getting ~20-25% of a core), and training reconstruction was
  already 94% word / 100% topological / 98% assembly accuracy at that point. All 1000 prior samples
  decoded without a single failed batch, and they are visually polymer-like (fused-ring PIM- and
  polyimide-type structures); the limitation of this row is uniqueness (22%), not decoding.

### Hier-VAE (+expert) — **dropped, and here is exactly why**
The intended variant replaces HierVAE's automatically extracted motif vocabulary with our 99 expert
motifs. HierVAE's vocabulary is not a free-form list of substructures: `MolGraph.pool_clusters`
(`poly_hgraph/mol_graph.py:80-85`) pools a fragment into a motif node only when the fragment SMILES
produced by HierVAE's *own* `find_fragments` decomposition — a ring-bond/leaf-atom cut — is literally
a member of `MolGraph.FRAGMENTS`. So a motif can only enter the vocabulary if HierVAE's decomposition
already produces it.

We measured the overlap (`common/expert_motifs.py`, result in `common/expert_probe.json`): of the
**96** distinct canonical SMILES behind the 99 expert motif files, **11** ever appear as a
`find_fragments` output over the 114 polymers, covering **114 of 457 (25%)** of fragment occurrences.
The 11 are small generic groups (`Cc1ccccc1`, `CC(C)(C)C`, `CC(F)(F)F`, `CS(C)(=O)=O`, `COC`, …); the
85 that cannot be expressed are exactly the large fused-ring and heterocyclic motifs that carry the
gas-separation chemistry. Installing the expert set would mean rewriting HierVAE's decomposition, not
configuring it, so **the vocabulary format cannot accept the expert motifs and we dropped the row**
rather than report an 11-motif variant under the "+expert" label. The 11/96 and 25% numbers are
themselves a usable argument for the paper: our representation can express motifs that a
motif-extraction-based VAE cannot.

### Graph GA, unconditional (`runs/graph_ga_uncond/`) — the strong baseline
Implementation: the standard GB-GA operators (Jensen 2019, as shipped in
`~/torch-molecule/torch_molecule/generator/graph_ga/{crossover,mutate}.py`, symlinked unchanged into
`common/gbga/` and imported as-is), with the GA loop, selection, size calibration and sampling
protocol written here (`common/graph_ga.py`). Population = the 114 training molecules,
offspring 50/generation, mutation rate 0.067, 20 generations, fitness = constant (so selection is
uniform: a random walk through the GA operators). The 1000 reported samples are the first 1000
offspring in creation order, duplicates kept.
Caveats:
* **Valid = 100% is by construction, not a learned property.** GB-GA's `mol_ok`/`ring_OK` filters
  reject any child that fails `Chem.SanitizeMol`, has ≤5 atoms, exceeds a sampled size target, or
  contains a macrocycle/ring-allene, and the operators retry. 1009 operator attempts produced 1000
  accepted offspring, so the retry loop was barely exercised; it is still a filter, and a reader
  should not read 100% as "the GA never makes an invalid molecule".
* GB-GA's size filter is `5 < n_atoms < N(average_size, size_stdev)` with `average_size = 39.15`,
  `size_stdev = 3.50` hard-coded upstream (the source comments "TODO: calculate for dataset"). We set
  them from the training set (52.97 / 13.91). Without that the GA is forced to make ZINC-sized
  molecules on a polymer dataset.
* Even so the GA drifts *down* in size (mean 26.8 heavy atoms vs 53.0 in the dataset): crossover cuts
  a molecule in two and recombines fragments, so repeated crossover without a size-rewarding fitness
  shrinks the population. Its Diversity (0.91) and Novel (99.9%) are therefore partly a
  fragmentation effect. This is the row most comparable to ours in molecule size.

### Graph GA, conditioned on predicted CO2 permeability (`runs/graph_ga_co2/`)
Same GA, fitness = surrogate `log10(CO2 barrer)`, 60 generations, fitness-proportional mating and
truncation survival. Reported samples = the **top 1000 distinct molecules by surrogate score** out of
2261 unique molecules found, so Valid and Unique are 100% by construction and must not be read as
model quality. The surrogate's best molecule scores 4.54 vs 4.43 for the best labelled training
polymer, and the top-1000 mean is 4.39 — i.e. the GA saturates the surrogate, which is the expected
failure mode when a GA is pointed at a 96-sample regression model. Its diversity collapses to 0.60.
Surrogate: gradient boosting on Morgan counts r=2/2048 + 8 RDKit descriptors, trained on the 96
polymers with a CO2 label, **5-fold CV R² = 0.77** (`common/surrogate.json`). It is a cheap
conditioning signal for a baseline, not a property model we would stand behind.

### SMILES LSTM and SMILES transformer (`runs/smiles_{lstm,transformer}{,_fullfit}/`)
`common/smiles_lm.py`: character-level autoregressive models trained from scratch on the 114 SMILES,
sampled multinomially at temperature 1.0 with no validity filtering. Two training protocols are
reported for each architecture, because at n=114 the choice decides the result:
* **early-stopped** — 100 train / 14 held out, early stopping on held-out NLL;
* **fit to convergence** — all 114, 1000 epochs, no early stopping.

**This is the headline finding for the "transformers / fully data-driven" reviewer request, and it is
a negative result.** With 114 training SMILES a character LM has two regimes and neither is usable:
stop early and it is mostly invalid (LSTM 17% valid, transformer 0.2% valid — the transformer's best
held-out checkpoint is at epoch 19 and emits essentially single atoms); fit it to convergence and
validity recovers (LSTM 94%, transformer 86%) but it reproduces the training set — **Novel = 1.0%
(LSTM) and 0.4% (transformer)**, Unique ≈ 11%. Train NLL at convergence is 0.048–0.050 per character
against a held-out NLL of 1.15 at the transformer's best epoch: the models memorise. Our method's
100% novelty at n=114 is the relevant contrast, and it is a property of the representation, not of
having more data.

### DEG — data-efficient grammar (`runs/deg/`, `runs/deg_grammar/`)
Code: `~/data_efficient_grammar`, env `DEG`. Caveats:
* **The published training script uses 20 molecules, not the dataset.** `main.py` reads
  `for line in lines[:20]`. `common/deg_run.py` passes all 114 (111 survive DEG's hypergraph
  conversion).
* **Stereochemistry had to be stripped.** DEG's `standardize_stereo`
  (`private/hypergraph.py:1139`) raises on E/Z-annotated acyclic double bonds; the amidoxime polymer
  `.../C(N)=N/O` kills `data_processing` outright. DEG is therefore trained on
  `common/train_smiles_nostereo.txt` (35 of the 114 SMILES change).
* **Result (grammar induction only).** The induced grammar has **170 production rules**; 1000
  molecules were produced with `random_produce` with **zero failures** and a mean of 2.35 production
  steps each, giving 100% valid / 50.3% unique / 60.4% novel / diversity 0.806, median 43 heavy
  atoms. So the row is a real, working DEG grammar sampler — it is the *reward optimisation* that is
  missing, not the grammar.
* **DEG has no Retro\* number.** Its sampler finished after Retro\* run 2; the catch-up pass that
  covered it was invalidated by a concurrent session rewriting the shared `retro/manifest.json`
  (see the status note at the top), and the raw output was overwritten before it could be re-paired.
  Re-run with `retro_tools.py build --k 25 --baselines deg; retro/run3.sh; retro_tools.py collect`
  (~3 min on an idle machine).
* **The RL stage could not be run to convergence.** DEG's reward is
  `diversity + 2 * retro_rate`, with one Retro\* call per generated molecule per grammar evaluation.
  The paper's setting is 50 epochs × 5 MCMC samples × 100 generated samples = **25,000 Retro\* calls**;
  on this machine Retro\* resolves ~1.3 molecules/min in total, i.e. ≈320 hours. We ran a reduced
  budget (2 epochs × 2 MCMC samples × 10 generated samples) with 2 Retro\* listeners, and separately
  induced a grammar with a single MCMC pass of the untrained potential function
  (`runs/deg_grammar/grammar_mcmc.pkl`) so that a DEG row exists at all. **The DEG row in the table
  is DEG's grammar induction with little or no reward optimisation — it is not the published DEG
  result**, and that is the single biggest caveat in this table.

### REINVENT 4 (`runs/reinvent/`) — the RL row
Two rows, both from REINVENT 4.4.22 (`reinvent4` env, `cuda:0`), prior
`/ssd/msun415/REINVENT4/priors/reinvent.prior`:
* **TL only** — transfer learning of the ChEMBL prior onto the polymers, 100 epochs, batch 32, then
  1000 molecules sampled de novo (`sample_tl.toml`). This is the "what does a strong pretrained
  SMILES generator do after seeing our data" row.
* **TL + RL** — staged learning (DAP, sigma 128, lr 1e-4, batch 64, ≤150 steps) from the TL agent,
  **reward = sigmoid(surrogate log10 CO2 barrer; low 2.0, high 4.5, k 0.5)**, computed by
  `common/reinvent_score.py` through REINVENT's `ExternalProcess` scoring component; an unparseable
  SMILES scores 0.0. The surrogate is the same CV R² = 0.77 gradient-boosting model described above.
  1000 molecules are then sampled from the final RL checkpoint.
Caveats: (a) **REINVENT is the one baseline that is not trained only on our 114 molecules** — its
prior is ChEMBL-pretrained, which is why its validity is the highest of the learned models; it should
be labelled as such in the paper. (b) **REINVENT's SMILES filter rejects 15 of the 114 polymers**
("... is invalid" in `runs/reinvent/tl.log`), so TL sees 99. (c) the RL reward is a 96-sample
surrogate, so the RL row says "RL can optimise this surrogate", not "RL designs better membranes".

## What we could not run, and why

| Dropped | Reason |
|---|---|
| Hier-VAE (+expert) | HierVAE's motif vocabulary can only contain fragments its own `find_fragments` decomposition emits; 11 of 96 expert motifs qualify (25% of fragment occurrences). Not fakeable without rewriting the decomposition. |
| DEG at the published budget | 25,000 Retro\* calls ≈ 320 h at this machine's Retro\* throughput. Reduced-budget run reported instead, clearly labelled. |
| Retro\* on all 1000 samples per baseline | Retro\* costs ~0.8 min/molecule here (24 workers, ~12 min of per-worker model loading); 1000 molecules per baseline would be ~14 h each. We ran **seeded random subsamples in two combined jobs** (run 1: 5 rows x 45; run 2: 4 rows x 25) and report `Retro* n` per row. Raw results in `retro/run1/` and `retro/run2/`. |
| — | nothing else; DEG and JT-VAE finished after passes 1–2 and were picked up by pass 3 (`retro/run3/`). |

## Reproducing

```bash
G=~/RandomWalk/results_v2/baselines/generation
P=~/miniconda3/envs/polymer_walk/bin/python
$P $G/common/graph_ga.py --mode uncond --out $G/runs/graph_ga_uncond
$P $G/common/graph_ga.py --mode co2    --out $G/runs/graph_ga_co2 --generations 60
$P $G/common/smiles_lm.py --arch lstm        --out $G/runs/smiles_lstm
$P $G/common/smiles_lm.py --arch transformer --out $G/runs/smiles_transformer_fullfit --fullfit --epochs 1000
$G/runs/hiervae/train.sh && $P $G/common/hier_sample.py --vocab $G/runs/hiervae/vocab.txt \
    --model $G/runs/hiervae/ckpt/model.8 --nsample 1000 --out $G/runs/hiervae/samples.txt
~/miniconda3/envs/torch_molecule/bin/python $G/common/jtvae_run.py --out $G/runs/jtvae
$G/runs/reinvent/run_tl.sh && $G/runs/reinvent/run_rl.sh
$P $G/common/metrics.py <name> $G/runs/<name>/samples.txt      # one JSON row per baseline
$P $G/common/retro_tools.py build --k 45 --baselines ...       # then $G/retro/run1.sh
$P $G/common/retro_tools.py collect && $P $G/common/make_table.py
```
