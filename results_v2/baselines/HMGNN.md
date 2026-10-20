# HM-GNN and HM-GNN (+expert) on the corrected polymer dataset

Port of *Molecular Representation Learning via Heterogeneous Motif Graph Neural Networks*
(Yu & Gao, ICML 2022) to the corrected `results_v2` data, under the honest protocol in
`baselines/bl_data.py`. All code and scratch work lives in `baselines/hmgnn/`. Twelve rows were
appended to `baselines/outputs/results.jsonl`: `HM-GNN`, `HM-GNN (+expert)` and — following the
convention already used by every other baseline in that file — `HM-GNN [raw target]`,
`HM-GNN (+expert) [raw target]`, three gas pairs each.

```
baselines/hmgnn/
  prep.py             rebuilds every HM-GNN input from results_v2/data/group-contrib and
                      constructs the heterogeneous-motif graphs (auto / autotyped / expert)
  train.py            training + the honest protocol + the result rows
  inspect_vocab.py    prints the motif vocabulary HM-GNN extracts by itself
  summarize.py        reads back the rows it wrote
  utils/              copy of ~/my-HM-GNN/utils (ops.py / gin.py / graphcnn.py / data_loader.py),
                      with one one-line bug fix, marked "PORT FIX" in data_loader.py
  HM_GNN_data/<pair>/ regenerated repo-format inputs (A.txt, node_labels.txt, edge_labels.txt,
                      graph_indicator.txt, graph_labels.txt, <pair>.txt)
  preprocessed_datasets/<pair>_{auto,autotyped,expert}.pkl   the pickled DGL HM graphs
  venv/               venv over the hm-gnn conda env, adding only dgl-cu113==0.9.0
  logs/               full stdout of all 216 config-runs
```

## Results

Normalized MAE and R^2 on the held-out test split, mean +- population std over split seeds 1,2,3.
Index 0 = permeability, index 1 = selectivity, exactly as in the result rows.

**With `log_target` in the grid (the `HM-GNN` / `HM-GNN (+expert)` rows).** `log_target=True` was
selected on validation in **18 of 18** runs.

| baseline | target | MAE | R^2 | per-seed R^2 (permeability) |
|---|---|---|---|---|
| HM-GNN | H2 | 0.719 +- 0.293 | **-3.231 +- 2.689** | -3.88, -6.15, 0.34 |
| HM-GNN | H2/N2 | 0.277 +- 0.053 | 0.865 +- 0.052 | |
| HM-GNN | O2 | 0.535 +- 0.285 | 0.040 +- 0.194 | -0.19, 0.02, 0.29 |
| HM-GNN | O2/N2 | 0.536 +- 0.101 | 0.608 +- 0.213 | |
| HM-GNN | CO2 | 1.009 +- 0.386 | **-13.78 +- 18.09** | -0.49, -39.36, -1.50 |
| HM-GNN | CO2/CH4 | 0.389 +- 0.105 | 0.645 +- 0.181 | |
| HM-GNN (+expert) | H2 | 0.328 +- 0.046 | 0.122 +- 0.810 | 0.78, -1.02, 0.61 |
| HM-GNN (+expert) | H2/N2 | 0.631 +- 0.200 | -0.368 +- 1.440 | |
| HM-GNN (+expert) | O2 | 0.982 +- 0.763 | **-33.37 +- 47.70** | 0.28, -100.8, 0.43 |
| HM-GNN (+expert) | O2/N2 | 0.613 +- 0.203 | 0.577 +- 0.246 | |
| HM-GNN (+expert) | CO2 | 0.626 +- 0.362 | **-6.182 +- 9.524** | 0.66, -19.65, 0.45 |
| HM-GNN (+expert) | CO2/CH4 | 0.378 +- 0.119 | 0.690 +- 0.046 | |

**Raw target only (the `[raw target]` rows), i.e. the same deal our own model gets.** These are the
numbers I would put in the paper; see "the log-target trap" below for why.

| baseline | target | MAE | R^2 |
|---|---|---|---|
| HM-GNN [raw target] | H2 | 0.498 +- 0.221 | 0.462 +- 0.143 |
| HM-GNN [raw target] | H2/N2 | 0.584 +- 0.185 | 0.417 +- 0.155 |
| HM-GNN [raw target] | O2 | 0.488 +- 0.246 | 0.281 +- 0.182 |
| HM-GNN [raw target] | O2/N2 | 0.567 +- 0.119 | 0.603 +- 0.086 |
| HM-GNN [raw target] | CO2 | 0.588 +- 0.344 | 0.226 +- 0.235 |
| HM-GNN [raw target] | CO2/CH4 | 0.418 +- 0.150 | 0.637 +- 0.093 |
| HM-GNN (+expert) [raw target] | H2 | 0.483 +- 0.099 | -0.526 +- 1.405 |
| HM-GNN (+expert) [raw target] | H2/N2 | 0.582 +- 0.285 | 0.523 +- 0.157 |
| HM-GNN (+expert) [raw target] | O2 | 0.377 +- 0.131 | 0.078 +- 0.289 |
| HM-GNN (+expert) [raw target] | O2/N2 | 0.630 +- 0.165 | 0.517 +- 0.254 |
| HM-GNN (+expert) [raw target] | CO2 | 0.612 +- 0.233 | -3.414 +- 5.249 |
| HM-GNN (+expert) [raw target] | CO2/CH4 | 0.473 +- 0.087 | 0.475 +- 0.212 |

Context (same splits, same metric): our model 0.47/0.48, 0.50/0.70, 0.35/0.51, 0.62/0.54,
0.37/0.49, 0.43/0.75; XGBoost on expert motif counts, permeability only, 0.301/0.681 (H2),
0.262/0.764 (O2), 0.316/0.718 (CO2); the training-mean reference 0.71/0.66/0.67 MAE at R^2 ~ -0.05.

**Summary.** HM-GNN is a weak-to-middling baseline on this dataset in every presentation.
`HM-GNN [raw target]` clears the training-mean reference on both targets for all three gas pairs
(positive R^2 everywhere, lower MAE everywhere), but its
permeability R^2 (0.46 / 0.28 / 0.23) is below our model (0.48 / 0.51 / 0.49) and far below XGBoost
on expert motif counts (0.68 / 0.76 / 0.72), and its selectivity is comparable to ours on O2/N2 and
CO2/CH4 and clearly worse on H2/N2. The expert motif vocabulary gives a lower *validation* loss than
the vocabulary the method extracts for itself in 51 of the 72 matched (pair, seed, config) slots,
and the gap at the selected config is large (H2 seed 1: 0.29 vs 0.44; CO2 seed 1: 0.24 vs 0.67;
CO2 seed 3: 0.13 vs 0.20). The best single runs come from it (expert H2 seed 1: MAE 0.319/0.348,
R^2 0.78/0.61 — on a par with XGBoost on expert motif counts),
but it does not translate into a consistently better selected test score, because one seed in three
is ruined by the exponentiation blowup described next and because selection on 15 validation points
is very noisy. Nothing here threatens the paper's claims.

## The log-target trap (this is the thing that affects how the comparison is presented)

`log_target` is in the grid because the other baselines get it, and validation picks it every single
time (18/18) — it is by far the best thing you can do for validation loss, often by 5x. But the
metric is MAE/R^2 on the *raw* target normalized by training statistics, and permeability spans four
orders of magnitude (1.4 to 33300 barrer for CO2). A neural net fitted in log space is free to
extrapolate above the largest training value; exponentiating a log-space error of ~1.5 sd on one
high-permeability test polymer produces a raw-space error of tens of sd, which R^2 squares.

The damage is always localized to one seed: HM-GNN (+expert) on O2 gets R^2 0.28, **-100.8**, 0.43
across seeds 1,2,3; on CO2, 0.66, **-19.65**, 0.45; HM-GNN on CO2, -0.49, **-39.36**, -1.50. The
offending seed is the one with the *lowest* validation MSE (0.11-0.21), so the protocol cannot catch
it. Note that the MAE columns are almost unaffected (MAE is linear, so one bad point costs ~1/19 of
the score) — it is specifically R^2 that explodes.

This is not something the XGBoost/RF/ridge baselines can suffer, because a tree or a kernel ridge
fit cannot predict outside the convex hull of its training targets, which is why the XGBoost
expert-motif baseline picks log targets in 8 of 9 runs and still reports R^2 ~ 0.7. A GNN can and
does. So:

* reporting a mean R^2 of -33 for HM-GNN (+expert) on O2 is arithmetically correct and completely
  uninformative about the model;
* the `[raw target]` rows are the fair HM-GNN number and are what I would quote; the log rows are
  there because the protocol asks for the option and because a reader should see what it does;
* if any table shows mean R^2 over seeds for log-target neural baselines, it should either show the
  per-seed values or a median, or say in the caption that R^2 here is not a robust statistic on
  19-point test splits with a four-decade target.

## The expert variant was possible

`preprocess.py` in the local repo already had an unused `-motif-file` hook and `utils/ops.py`
already had a `GenGraph(data, motif_file=...)` branch that replaces `nx.cycle_basis` with an
externally supplied atom partition, so the method does accept an external vocabulary in principle. I
did not use that hook, because it wants a per-molecule *atom-index* partition, which would have
meant reconstructing the atom-level mapping from each polymer SMILES back onto its motif walk.

That reconstruction is unnecessary. The HM graph is bipartite: molecule nodes on one side, motif
nodes on the other, with

* edge weight `(count of motif m in molecule i / number of motif slots in molecule i) * log((N+1)/df(m))`
  (`GenGraph.gen_components` + `GenGraph.update_weight`), and
* molecule node features = the binary indicator of which motif nodes it touches
  (`preprocess.py: gen_features_labels`).

Both depend on the molecule's motif *multiset* and nothing else, so an expert HM graph is fully
determined by each polymer's expert-motif multiset. I read that straight off the `[PSL]\d+` labels
in `polymer_walks_v2_preprocess.txt`, which is the ground-truth decomposition, and rebuilt the same
bipartite graph with the same weighting formula (`prep.py: build_expert`). The arithmetic is
identical to the shipped code; only the clique set per molecule changes.

Of the 99 curated motifs in `all_groups/`, only **55** ever occur in the 114 polymers; after the
per-gas-pair mask, 51 (H2, O2) and 49 (CO2) remain, and those are the motif nodes. This matches
`bl_data.feat('motif_expert')`, so HM-GNN (+expert) and the Bag-of-Motifs (+expert) baseline see the
same vocabulary.

Everything else is unchanged between the two variants: `TwoGIN` still concatenates the HM-graph GIN
with a 5-layer GIN over the atom graph of the repeat unit, and the atom branch is identical in both.

## What the automatic vocabulary actually contains (also important for the comparison)

`inspect_vocab.py` dumps it. HM-GNN's own construction enumerates, per molecule, every ring of
`nx.cycle_basis` plus every bond that is in no ring, and identifies each by its atom labels (up to
rotation) together with the bond "weights". The repo loads bond labels only when `-edge_weight` is
passed, and it defaults to `False` — neither the authors' `run.sh`/`run_gc.sh` nor anything else in
the repo ever passes it, and the only TUDataset directories that even ship an `edge_labels.txt` are
unused by those scripts. With all weights 1, the entire vocabulary on this dataset is **21 motifs**
(19 for the CO2 mask):

```
C-C-C-C-C-C   555      <- any six-membered carbocycle: benzene and cyclohexane are the same motif
C-C           539
C-O           379
C-F           216
C-C-C-C-N     130
C-N           108
C-C-C-C-C     102
C-C-O-C-C-O    63
C-C-C-N-C-N    32
...  (12 more, all with <= 12 occurrences)
```

So the "motifs" HM-GNN extracts here are an element-pair bond census plus a ring census typed only
by ring size and ring elements. They carry almost no chemistry: the construction cannot tell benzene
from cyclohexane, a nitrile from an amine, or an ester from an ether. That, not the architecture, is
why the default variant's HM-graph branch is so much weaker than the expert one. For reference I
also built the bond-typed graph (`-edge_weight`, 31 motifs: aromatic C6 separates from saturated C6,
C=O from C-O, nitrile C#N from C-N) and left it in
`preprocessed_datasets/<pair>_autotyped.pkl`; it is **not** part of the reported numbers, because
the shipped default and the authors' own scripts use the untyped construction and the compute budget
did not stretch to putting both constructions in the grid. It would be the first thing to try if
anyone wants to strengthen this baseline.

## The protocol

`train.py` imports the shared harness and writes no splitting code of its own:

* splits: `bl_data.split(pair, seed)` for seeds 1, 2, 3 (N = 94, 94, 92; 60/15/19 and 58/15/19).
* targets: both predicted **jointly** by one model with a 2-dimensional output. `main.py`'s
  regression path already takes `num_classes = labels.shape[1]` and an L1 loss over all columns, so
  this is the shipped behaviour, not a change. Reported MAE/R^2 are per target, index 0 =
  permeability, index 1 = selectivity.
* normalization: `bl_data.normalize(y, train_idx)`, i.e. **training-split** mean/std.
* metrics: MAE and R^2 on the raw targets normalized by training-split statistics, so they line up
  with the `heldout_*` columns of our own runs. When a config uses `log_target`, the model is fit on
  the training-normalized log target, predictions are de-normalized, exponentiated, and only then
  re-normalized with the raw training statistics and scored (same `np.clip` overflow guard as
  `bl_classical.py`, no range clipping). Validation MSE is computed on that same normalized-raw
  scale for every config, so log and non-log configs are compared on the same quantity.
* selection: for each (pair, seed) independently, the config with the lowest validation MSE wins,
  and within a run the epoch with the lowest validation MSE wins. The test split is only ever
  evaluated — never used for epoch choice, config choice, early stopping or normalization. This
  mirrors `bl_classical.py`, which is why `per_seed[].cfg` and `per_seed[].log_target` differ by
  seed.
* grid: `lr` in {0.05, 0.01} x `h_dim` in {16, 64} x `log_target` in {False, True} = 8 configs
  (4 for the `[raw target]` rows); `l_num=4`, `drop_n=drop_c=0.2`, `w_d=5e-4`, full-split batch,
  Adam, L1 loss, all at the repo defaults. 216 config-runs in total. `best_epoch` is recorded in
  `per_seed[].cfg`.

What the shipped `main.py` did instead, and which this replaces: `random.shuffle` of all indices,
80% train / 20% validation, **no test set at all**, and the reported score was
`min over epochs of the validation MAE` — the best of 2000 validation evaluations, reported as the
model's performance. On this data that is worth a large factor: the selected runs reach validation
MSE 0.07-0.3 while the test MSE at the same epoch is several times larger (HM-GNN H2 seed 2:
validation 0.071, test R^2 -6.1).

## What I had to change, and why

1. **All preprocessed artifacts were regenerated.** `preprocessed_datasets/smiles_and_props_*`,
   `dataset_init/*` and `HM_GNN_data/*` in `~/my-HM-GNN` were built from
   `/home/msun415/polymer_walk/datasets/group-contrib-smiles/`, which no longer exists, and from the
   earlier broken dataset. `prep.py` rebuilds the repo-format inputs directly from
   `results_v2/data/group-contrib` with RDKit, in `bl_data.masked(pair)` order, and then runs the
   repo's own `FileLoader` + `GenGraph` on them. I did not run `preprocess_init.py` itself, because
   it only exists to turn SMILES into those five text files via a PyTorch-Geometric dataset class
   whose input path is gone and whose label handling is the leak in item 3; `prep.py` writes exactly
   the same five files (node label = atomic number, node tag = atomic number - 1, A.txt in both
   directions, local 0-based neighbour lists), which is what `preprocess.py` consumed.
   Nothing under `~/my-HM-GNN` was modified.
2. **The hardcoded `number_of_graphs = 94`** in `main.py` is gone; the molecule count comes from the
   regenerated data (94 / 94 / 92, which happens to agree for H2 and O2).
3. **Label normalization leak fixed.** `dataset_init.py` normalized with the mean/std of *all*
   polymers (`labels = (labels - np.mean(labels, 0)) / np.std(labels, 0)`, before any split). The
   regenerated `graph_labels.txt` holds raw barrer / unitless values, and nothing downstream reads
   them: `train.py` takes targets from `bl_data.targets(pair)` and normalizes per split with
   `bl_data.normalize(y, train_idx)`.
4. **Deduplication: I did not dedup, deliberately.** The reported bug is real —
   `if not (line.strip() in smiles_list)` compares a whole `"<smiles> <p> <s>"` line against a list
   that only ever receives `line.split()[0]`, so the condition is always true and nothing is ever
   removed. 3 of the 114 monomer SMILES do occur twice (111 unique canonical SMILES) — two polyimide
   pairs and one spiro analogue. Not deduping is the right call, and `n` stays at 94/94/92 for every
   baseline.

   **Correction (checked against the data after this port was written).** Two claims in the original
   version of this note were wrong, and the conclusion is stronger than it said:
   - The duplicate pairs do **not** have different expert motif walks. `PI-TB-3`/`TPDA2-6FDA-PI`,
     `PI-TB-4`/`TPDA2-ODPA-PI` and `6FDA-CTBDA`/`6FDA-iCTBDA` each share a *byte-identical* walk
     string, so the expert motif branch cannot distinguish them either, and neither can our own model.
   - They do **not** have different measured permeabilities, because in each pair only one member has
     any measurements at all: `TPDA2-6FDA-PI`, `TPDA2-ODPA-PI`, `6FDA-CTBDA` and `6FDA-iCTBDA` are all
     `(-,-,-,-,-)`. The per-gas-pair mask requires both target columns, so **no duplicate pair ever
     co-occurs in a masked set**: the H2/O2/CO2 masks keep `PI-TB-3` and `PI-TB-4` only, and neither
     member of the third pair.

   So there is no label-noise floor, for this port or for any other baseline, and the dedup question
   is moot rather than a judgement call. Verify with:
   `python -c "import bl_data as D; print([r['name'] for r in D.masked('H2')])"`.
5. **One-line bug fix in `utils/data_loader.py`** (marked `PORT FIX`). It read node labels as
   `int(i[-2])` — the second-to-last character of the line — which silently truncates any label
   >= 10. On this dataset that maps Na(11) -> 1, S(16) -> 6 (i.e. to carbon) and Br(35) -> 5, so
   sulfur-containing rings and bonds were merged into carbon motifs in the automatic vocabulary.
   Fixed to `int(i.strip())`. Affects only the automatic vocabulary, not the expert one.
6. **`GraphCNN`'s fixed sparse operands are cached** (`train.py: CachedGraphCNN`) and the
   full-neighbour sampler output is materialized once per split instead of per epoch. Both are pure
   speedups: identical arithmetic, no change to the model or the graph. Necessary because the host
   was running another campaign at load average ~140.
7. **A venv for CUDA DGL.** The `hm-gnn` conda env has a CPU-only DGL build (`g.to('cuda')` raises
   `Device API gpu is not enabled`), so the GPU could not be used at all as shipped. I left that env
   untouched and created `baselines/hmgnn/venv`, a venv over it with `--system-site-packages` whose
   only extra package is `dgl-cu113==0.9.0`. Same torch 1.11.0+cu113, same everything else. In the
   end most runs went on CPU anyway: the A6000 was busy with another job, and with it contended a
   CPU process was as fast per epoch (0.46 s) as a GPU one.
8. **Device plumbing.** `TwoGIN` and `GraphCNN` hardcode `self.device = torch.device('cpu')`, which
   only controls where their sparse operands are placed; `train.py` sets it from `-device` after
   construction.

Scheduling note: the 216 config-runs were spread over at most six (briefly twelve) single-threaded
processes, one per (baseline, gas pair), each with `OMP_NUM_THREADS=1` — a handful of 128 cores.

## Caveats

* **R^2 is not robust here.** See "the log-target trap". One test polymer out of 19 can take mean
  R^2 from +0.4 to -100. The MAE columns are the stable ones.
* **HM-GNN is transductive, and that favours it.** The HM graph is built over all masked polymers at
  once; the motif vocabulary, the document frequencies in the IDF term, and the graph a test
  molecule sits in are all computed from the full set, and a test molecule's HM-graph embedding
  aggregates over motif nodes that aggregate over training molecules. No test *label* is ever
  touched — the leak-free part of the protocol holds — but this is not an inductive model, whereas
  the feature-based baselines (XGBoost on motif counts, Morgan + RF, polyBERT + ridge) are strictly
  inductive. If the comparison is presented as like-for-like, that difference should be stated.
  Making HM-GNN inductive would require rebuilding the graph per split, which is a change to the
  method, not to the protocol.
* **Epoch budget.** 400 epochs per run, not the shipped 2000, because the machine was saturated. An
  earlier 200-epoch pass was discarded before any row was written because it was selecting epochs in
  the 130-195 range, i.e. at its cap. At 400 the selected epochs are spread over 25-398 but still
  skew late (12 of 18 log-target runs selected an epoch past 250, two of them at 393 and 398), so
  these numbers are if anything a mild *under*-estimate of HM-GNN. A 2000-epoch grid is the obvious
  thing to run if the baseline is challenged.
* **Narrow grid.** 8 configs, versus 54 for the XGBoost baselines; `n_configs` in the rows records
  it. `lr` and `log_target` move validation loss a lot here, `h_dim` barely moves validation loss but
  does move test error, and the shipped `lr=0.05` default is bad (validation MSE ~1.5-1.7 against
  ~0.1-0.9 for `lr=0.01`), so anyone rerunning `main.py` at its defaults will get much worse numbers
  than these. A wider grid would mostly add selection noise.
* **Tiny, high-variance splits.** 15 validation and 19 test polymers. The per-split variance of the
  normalized targets ranges from 0.39 to 4.96 across seeds and columns, which is why a config can
  win validation and lose test, and why every baseline on this dataset has a large std over seeds.
* **Only three seeds**, as specified, so a single pathological seed is a third of the mean.
