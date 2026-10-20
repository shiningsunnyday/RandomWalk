# Generation baselines for the polymer-design resubmission — PLAN

Written 2026-10-19 22:07 EDT, **before** any training. Everything below lives under
`~/RandomWalk/results_v2/baselines/generation/`. Nothing outside this directory is written to.

Reviewer ask: comparisons against "VAEs, RL-based molecular design, transformers, fully data-driven
generative models, and existing polymer design methods, both ML-based and heuristic", on the same
dataset, with the same metrics.

## Dataset (read-only)

* `~/RandomWalk/results_v2/data/group-contrib/polymer_walks_v2_smiles.txt` — 114 polymer repeat units,
  `SMILES<space>name` per line. This is the **corrected v2** dataset; it is not the one used in the
  earlier version of this work.
* `~/RandomWalk/results_v2/data/group-contrib/polymer_walks_v2_preprocess.txt` — motif-walk encoding +
  `(H2,N2,O2,CH4,CO2)` permeabilities in barrer, `-` = missing. Used only for the surrogate reward /
  GA fitness, not for the unconditional baselines.
* `~/RandomWalk/results_v2/data/group-contrib/all_groups/{1..99}.mol`, `names.txt` — the 99 expert motifs.

Every baseline is trained on **exactly the same 114 SMILES**, no external corpus, no pretrained
chemistry checkpoints (except where noted in RESULTS.md). All 114 are used for training: with n=114 a
held-out split would make the generative models even weaker and the metrics here (valid/unique/novel/
diversity) do not need a test split. Novelty is measured against all 114, so "novel" means "not a
training molecule" for every method, ours included.

## Baselines, in priority order

| # | Baseline | Family (reviewer bucket) | Code | Trained on |
|---|---|---|---|---|
| 1 | **JT-VAE** | VAE / data-driven | `~/torch-molecule` (`torch_molecule.generator.jtvae`, ICML'18 JT-VAE port) | 114 SMILES |
| 2 | **Hier-VAE** | VAE, motif-based; the paper's existing baseline | `~/hgraph2graph/polymers` (HierVAE, ICML'20) | 114 SMILES, auto-extracted motif vocab |
| 3 | **Hier-VAE (+expert)** | motif VAE with *our* expert vocabulary | same | 114 SMILES, vocab forced to the 99 expert motifs |
| 4 | **Graph GA (uncond.)** | heuristic / non-ML polymer design | written here (`graph_ga.py`, RDKit) | 114 SMILES as initial population |
| 5 | **Graph GA (CO2)** | heuristic + surrogate objective | same + surrogate | 114 SMILES + CO2 labels |
| 6 | **SMILES LSTM / char-LM** | autoregressive, fully data-driven | written here (`smiles_lm.py`, PyTorch) | 114 SMILES (char level) |
| 7 | **SMILES transformer** | transformer | written here (`smiles_lm.py --arch transformer`) | 114 SMILES (char level) |
| 8 | **DEG** | data-efficient grammar (ML-based polymer design) | `~/data_efficient_grammar`, env `DEG` | 114 SMILES |
| 9 | **REINVENT 4** | RL-based molecular design | env `reinvent4` | RL against a surrogate reward; prior = REINVENT's own |

Stretch only if time remains: nothing else. 4–5 solid rows with honest caveats beats 9 broken ones.
A codebase that fights me for more than ~90 min gets a precise "could not run X because Y" paragraph
in RESULTS.md and is dropped.

Known risks, decided in advance:
* **Hier-VAE (+expert)**: HierVAE's `pool_clusters` only pools a fragment when the fragment SMILES
  produced by its own `find_fragments` decomposition is literally in `MolGraph.FRAGMENTS`. The expert
  motifs are R-site-annotated `.mol` files, a different convention. I will measure how many of the 99
  expert motifs actually appear as `find_fragments` outputs on the 114 molecules. If the coverage is
  negligible the vocabulary format cannot accept them, and I will say exactly that in RESULTS.md and
  drop the row rather than fake it.
* **REINVENT/RL** needs a reward; it will be a surrogate trained here on the 114 polymers, and the
  exact reward definition will be printed in RESULTS.md. If the surrogate is bad, the row is still
  reportable as long as the reward is stated.
* **DEG** is known-slow; it is last for that reason.

## Surrogate (used only by Graph GA (CO2) and REINVENT)

Trained here, in this directory, on the subset of the 114 polymers with a CO2 permeability label:
target = `log10(CO2 barrer)`, features = Morgan count fingerprint r=2, 2048 bits + 8 RDKit descriptors,
model = gradient boosting / random forest selected by 5-fold CV R^2. Its CV R^2 is reported in
RESULTS.md so a reader can judge how meaningful the conditioned row is. It is a *cheap* surrogate for
baseline conditioning, not a claim about our prediction results.

## Metrics — exact definitions implemented in `common/metrics.py`

1000 samples per baseline, RDKit, matching `~/RandomWalk/results_v2/compute_numbers.py` lines 18–37
line for line.

* `n_sampled = 1000` — raw strings emitted by the method, in sampling order, no filtering.
* **Valid** = #{s : `Chem.MolFromSmiles(s)` is not None and `Chem.SanitizeMol` succeeds} / 1000.
  (`MolFromSmiles` already sanitizes; `SanitizeMol` is called again explicitly so the definition is
  literally the one in the task.) Baselines emit plain SMILES, so there is no Walk/SMARTS analogue —
  those two columns are ours only and are left blank for baselines.
* canonicalization: `Chem.MolToSmiles(Chem.MolFromSmiles(s))`, applied to the valid ones -> list `can`
  (duplicates kept, sampling order kept).
* **Unique** = `len(set(can)) / 1000` (the task's denominator). Also recorded:
  `unique_of_valid = len(set(can)) / len(can)`, because for a baseline with low validity the /1000
  version mixes validity into uniqueness. RESULTS.md prints the /1000 number and footnotes the other.
* **Novel** = `sum(x not in train for x in can) / len(can)`, `train` = the 114 dataset SMILES
  canonicalized the same way (this is the reference implementation's denominator). Also recorded:
  `novel_of_unique`.
* **Diversity** = `1 - mean pairwise Tanimoto` over `AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)`
  on `can` (duplicates kept, exactly as the reference does). Also recorded: `diversity_unique` over
  `set(can)`.
* **Size** = `mean_heavy_atoms`, `median_heavy_atoms` (`mol.GetNumHeavyAtoms()`) and `mean_mw`
  (`Descriptors.MolWt`) over `can`.
* **Retro\*** = retrosynthesis success rate from
  `~/miniconda3/envs/retro_star_env/bin/python ~/RandomWalk/repro_v1/retro.py <in> <out> 24`
  on `can`, success = last field of the output line is `True`.
  `retro.py` only writes its output file after all workers finish, so for each baseline `can` is
  **shuffled with a fixed seed (0) and split into chunks of 100**, run chunk after chunk, one baseline
  at a time. The reported Retro\* is over all molecules in finished chunks, with `retro_n` recorded.
  A seeded shuffle makes a partial result an unbiased estimate instead of a prefix artefact.
* Also recorded per baseline for context: `frac_contains_dummy` (`*`/R-group stand-ins left in the
  output), wall-clock train and sample time, and the exact command run.

Outputs: `runs/<baseline>/samples.txt` (1000 raw lines), `runs/<baseline>/canon.txt`,
`runs/<baseline>/retro_in_*.txt`/`retro_out_*.txt`, `runs/<baseline>/log*.txt`, plus one JSON row per
baseline appended to `metrics.jsonl`.

## Machine budget

Load average is already ~128 from another campaign, so: every CPU job runs under `nice -n 10` with
`OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`, at most ~4 of my own CPU processes at a time, except the
Retro\* runs which use the prescribed 24 workers (also `nice -n 10`) and are run strictly one at a
time, last. All neural training goes on `cuda:0` (RTX A6000, idle), one job at a time.
Conda pythons are called by absolute path; the `python` on `PATH` is an unrelated python2.
No existing conda env is modified: extra packages, if needed, go into
`common/pylibs/` via `pip install --target` and are picked up with `PYTHONPATH`.

## Time budget (start 22:07 EDT 2026-10-19; hard stop for new work ~07:30 EDT)

| Slot | Work |
|---|---|
| 22:07–22:35 | PLAN.md, `common/metrics.py`, data prep, expert-motif coverage probe, Retro\* cost probe |
| 22:35–23:30 | Hier-VAE: vocab + preprocess + train (GPU) in background; meanwhile write Graph GA |
| 23:30–00:30 | Graph GA uncond. + surrogate + Graph GA (CO2); sample 1000 each |
| 00:30–01:30 | SMILES LSTM + transformer (GPU, minutes each), JT-VAE (GPU) |
| 01:30–02:30 | Hier-VAE (+expert) if the coverage probe says it is honest; otherwise drop and document |
| 02:30–04:00 | DEG (hard 90 min limit), then REINVENT 4 (hard 90 min limit) if DEG finishes early |
| 04:00–07:30 | Retro\* chunks, one baseline at a time, in priority order; metrics + RESULTS.md written incrementally so the deliverables are complete even if Retro\* is still running |

STATUS.md is kept current at every transition so another agent can take over.
