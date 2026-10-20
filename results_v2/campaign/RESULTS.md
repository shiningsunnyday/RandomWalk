# RESULTS — cross-split agreement campaign

Headline metric, protocol and arms: `PLAN.md` (written before any new-configuration run; two dated amendments,
the second of which reverts the first). Live state and what is still running: `STATUS.md`. What to adopt:
`RECOMMENDATION.md`.

Written 2026-10-20 03:05 EDT. **Ten random-split arms of 60 runs, the 72-run variance-replication arm and four
24-run hold-out arms are complete: 768 runs, every one 1000 epochs.** Two arms were still running
(`strat_struct`, `huber`) and `smooth` was queued; §8 lists them and says why none can change the conclusion.
`summarize_campaign.py` rescans everything and prints incomplete arms, the hold-out arms and `tseed` in a
separate block so they are never ranked against the random-split arms by mistake.

---

## 1. Headline table — all completed arms

K = 10 split seeds for every arm, same seeds 1–10, 1000 epochs, `ew = 0`, `init = normal`, directed message
passing except where stated; `fw` (grammar-W features) selected per gas pair on **validation** loss.

**V = mean over the 6 properties of the across-split population std of test R²** (lower is better), with a
2000-sample bootstrap CI over split seeds. R² is in the space the model was trained in; `rawR²` re-expresses it
in raw barrer space for every arm so the spaces can be compared. **`valρ` is the pre-committed arm-selection
rule** (mean validation Spearman, validation only).

| arm | what changed | **V** ↓ | V 95% CI | mean test R² | mean test MAE | mean test ρ | raw-space R² | **valρ** (rule) |
|---|---|---|---|---|---|---|---|---|
| frozen (reference, K = 3) | — | 0.198 | — | 0.580 | 0.456 | — | 0.580 | — |
| **`combo_y2`** | **log target + joint permeability×selectivity stratified split** | **0.136** | **[0.079, 0.156]** | **0.619** | 0.477 | **0.799** | 0.213 | **0.806** |
| `combo_y` | log + `val_smooth 25` + permeability-stratified | 0.155 | [0.091, 0.187] | 0.637 | 0.443 | 0.787 | 0.408 | 0.767 |
| `log` | `--log_target` only | 0.164 | [0.114, 0.180] | 0.602 | 0.465 | 0.750 | 0.322 | 0.785 |
| `combo_scaf` | log + `val_smooth 25` + scaffold strata | 0.180 | [0.121, 0.197] | 0.574 | 0.483 | 0.740 | −0.189 | 0.735 |
| `log_smooth` | log + `val_smooth 25` | 0.248 | [0.095, 0.333] | 0.557 | 0.470 | 0.751 | −0.242 | 0.730 |
| `base_undir` | undirected message passing | 0.410 | [0.229, 0.493] | 0.421 | 0.445 | — | 0.421 | 0.673 |
| `strat_y2` | joint permeability×selectivity strata, **raw** targets | 0.428 | [0.253, 0.500] | 0.386 | 0.461 | 0.726 | 0.386 | 0.754 |
| `strat_y` | permeability-tertile strata, raw targets | 0.435 | [0.194, 0.538] | 0.426 | 0.431 | 0.748 | 0.426 | 0.721 |
| `base` | **nothing — the frozen protocol at K = 10** | 0.516 | [0.225, 0.662] | 0.363 | 0.465 | — | 0.363 | 0.650 |
| `strat_scaf` | Murcko-scaffold-group strata (label-free), raw targets | 0.581 | [0.224, 0.719] | 0.309 | 0.460 | 0.687 | 0.309 | 0.685 |

**Paired differences against `base` on identical splits** (n = 60 = 3 pairs × 10 seeds × 2 properties):

| arm | Δ test R² | se | t | Δ raw-space test R² | se | t |
|---|---|---|---|---|---|---|
| `combo_y2` | **+0.256** | 0.076 | **3.35** | −0.150 | 0.147 | −1.02 |
| `log` | **+0.239** | 0.082 | **2.91** | −0.041 | 0.099 | −0.42 |
| `log_smooth` | +0.194 | 0.069 | 2.80 | −0.605 | 0.555 | −1.09 |
| `strat_y` | +0.063 | 0.054 | 1.16 | +0.063 | 0.054 | 1.16 |
| `base_undir` | +0.058 | 0.069 | 0.84 | +0.058 | 0.069 | 0.84 |
| `strat_y2` | +0.023 | 0.075 | 0.30 | +0.023 | 0.075 | 0.30 |
| `strat_scaf` | −0.054 | 0.104 | −0.52 | −0.054 | 0.104 | −0.52 |

**The factorial is unambiguous: the log target is the whole effect, and stratification does nothing without
it.** Every arm that changes only the split is statistically indistinguishable from `base`
(t = +1.16, +0.30, −0.52 for permeability-tertile, joint and scaffold strata), while every arm containing the
log target clears t = 2.8. Stratification pays only *on top of* the log target, and then only with the joint
permeability × selectivity strata: `combo_y2` − `log` test Spearman +0.048 ± 0.014 (t = 3.53).

`combo_y2` − `log`, paired: test R² +0.017 ± 0.025 (t = 0.68), test MAE +0.012 ± 0.015 (t = 0.85),
**test Spearman ρ +0.048 ± 0.014 (t = 3.53)**, raw-space R² −0.108 ± 0.124 (t = −0.87). So adding the
stratification on top of the log target buys a small but statistically solid gain in rank accuracy and about
17% off V, at no measurable cost in R² or MAE.

**Winner: `combo_y2`.** It is also the arm the pre-committed rule selects (highest validation Spearman, 0.806),
so the choice and the result agree. V falls from 0.516 to 0.136 — a **3.8× reduction**, far beyond the "25%
relative or it is not a result" bar set in `PLAN.md` §6, and the bootstrap intervals do not overlap.

Per-split and per-property numbers for every arm: `outputs/headline.txt`, `outputs/headline.json`,
`outputs/campaign_runs.csv`, `outputs/selected_<arm>.csv`, `outputs/table2_<arm>.txt`,
`outputs/val_spearman.csv`.

## 2. The frozen Table 2's error bars are far too small

This is the first thing the campaign established and it reframes the brief. Re-running the frozen protocol
**unchanged** at 10 split seeds instead of 3:

| property | frozen, K = 3 | `base`, K = 10 |
|---|---|---|
| H₂ | 0.48 ± 0.27 | 0.47 ± 0.22 |
| H₂/N₂ | 0.70 ± 0.08 | 0.33 ± **0.68** |
| O₂ | 0.51 ± 0.36 | 0.32 ± 0.31 |
| O₂/N₂ | 0.54 ± 0.13 | 0.49 ± 0.27 |
| CO₂ | 0.49 ± 0.34 | 0.31 ± **0.55** |
| CO₂/CH₄ | 0.75 ± 0.01 | 0.26 ± **1.07** |
| V | 0.198 | **0.516** |
| mean R² | 0.580 | 0.363 |

The failure mode is not uniform noise but **occasional catastrophic splits**: seven of ten splits sit in the
normal 0.4–0.9 band, while split seed 4 gives CO₂/CH₄ R² = −2.91 and H₂/N₂ R² = −1.50, and seeds 9–10 give
CO₂ R² = −0.47 and −0.88. The frozen three splits contained none of these. A 3-sample population standard
deviation cannot support the Table 2 error bars or the screening thresholds, independently of any modelling
change.

## 3. Why it happens — the R² denominator, not extrapolation

`analysis/why_variance.py` → `analysis/why_variance.txt`. R² = 1 − SSE/SST and SST is the **test split's own**
target variance. Across the nine frozen (pair, seed) cells, mean test R² correlates **+0.69** with the test
split's target spread relative to the training maximum (+0.35 with its log spread, +0.36 with its raw spread),
and the two collapsing cells are the narrowest test splits of their gas pair: H₂/N₂ seed 2 (smallest log spread
0.717, smallest raw spread 2019 → mean R² −1.03) and O₂/N₂ seed 2 (smallest raw spread 819 → −0.62). With ~19
test polymers, a split that happens to be narrow turns an ordinary absolute error into a large negative R².

This is amplified by the data: the permeability columns span 3.2–4.4 decades with skew +2.8 to +3.5 and the five
most permeable polymers carry **63–71%** of each column's total squared deviation (§5). Both fixes follow
directly — the log target compresses the range so SST varies far less from split to split, and stratifying on
target quantiles equalises SST by construction. That is exactly the ordering the arms show.

*(An earlier version of this section claimed the opposite mechanism — that the collapsing splits were the ones
containing an extrapolation point, correlation −0.56. It read `test_idx.txt` from the frozen run dirs, which is
corrupted; see §7. With the splits recomputed from the seed, no frozen test split contains a polymer more
permeable than the training maximum, the ratio being 0.43–0.67 in all nine cells, and every correlation flips
sign. The claim is withdrawn.)*

## 4. Where the variance comes from: 43–45% of it is not the split at all

`tseed`: the same three fixed splits repeated with four torch seeds, both `fw` settings, 72 runs.

| | within-split std over torch seeds | across-split std (torch-seed-averaged) | optimization share of the std budget |
|---|---|---|---|
| `fw = 0` | 0.118 | 0.156 | **43%** |
| `fw = 1` | 0.125 | 0.151 | **45%** |

Nearly half of what Table 2 reports as cross-split disagreement is **run-to-run optimization noise at a fixed
split**, not disagreement between splits. Two consequences: (a) averaging each split's metrics over a handful of
torch seeds, or ensembling torch seeds into each split's member, is a cheap and legitimate variance reduction
(with M = 4 the combined std falls from √(w²+a²) = 0.196 to √(w²/M + a²) = 0.167, ≈15%); (b) it explains why
`fw = 1` runs are not reproducible at all (§7.2) — the trajectory is chaotic, so a 1e-7 perturbation at epoch 1
reaches O(1) in test R² by epoch ~300.

## 5. Dataset-bias characterization (reviewer deliverable 1)

`analysis/dataset_bias.py` → `analysis/dataset_bias.json`, `analysis/figs/bias_property_hist.png`,
`bias_chemspace.png`, `bias_motif_usage.png`.

**Labels.** 114 polymers, but 18–22 are missing each gas, so a gas pair has 92–96 usable labels (94 for H₂/N₂
and O₂/N₂, 92 for CO₂/CH₄).

**Property distributions are extremely heavy-tailed.**

| property | n | span (decades) | skew | top-5 polymers' share of total squared deviation |
|---|---|---|---|---|
| H₂ | 94 | 3.2 | +2.8 | 0.63 |
| O₂ | 94 | 4.1 | +3.5 | 0.71 |
| CO₂ | 96 | 4.4 | +3.2 | 0.71 |
| H₂/N₂ | 94 | 1.6 | +3.0 | 0.68 |
| O₂/N₂ | 94 | 0.5 | +1.3 | 0.41 |
| CO₂/CH₄ | 92 | 1.8 | +2.3 | 0.59 |

**Composition.** 55 of the 99 motifs appear at all; the 10 most common cover **65%** of all motif slots
(P17 13%, P22 11%, S32 8%, P15 7%, L16 5%); 10 motifs appear exactly once. Median 8 motifs per polymer (3–19).

**Chemical space.** 79 unique Bemis–Murcko scaffolds, 60 of them singletons, largest 6 members. Mean pairwise
Tanimoto (Morgan r = 3, 2048 bits) 0.18 and median 0.13 — globally diverse — but median nearest-neighbour
Tanimoto 0.68 with 44% of polymers above 0.7. The set is diverse *in clusters*, which is why a random 80/20
split is easy: almost every test polymer has a close neighbour in training.

**The dataset is not uniformly high-free-volume**, which answers the reviewer's premise directly, but it is
unbalanced. The four fingerprint clusters differ by an order of magnitude:

| cluster | n | median H₂ | median O₂ | median CO₂ | median O₂/N₂ |
|---|---|---|---|---|---|
| 1 | 37 | 132 | 13.8 | 67 | 5.04 |
| 2 | 28 | 545 | 133 | 794 | 4.09 |
| 0 | 11 | 1295 | 320 | 2180 | 3.74 |
| 3 | 38 | 1368 | 503 | 2760 | 3.78 |

By name family the median H₂ permeability runs 134 (`polyimide (other)`, n = 5), 165 (`numbered entry`, n = 24),
501 (dianhydride-series polyimides, n = 26), 597 (PIM-PIs, n = 15), 1334 (PIMs, n = 33), 1625 (KAUST-PIs, n = 7).
No family exceeds 29% of the dataset.

**Honest summary for the SI:** the dataset spans a 10× range of permeability and is not monolithic, but the
high-permeability PIMs are the best-represented group, the selectivity range is narrow (O₂/N₂ half a decade),
five polymers out of ~95 carry two thirds of each property's variance, and the chemical space is clustered
rather than uniformly covered.

## 6. Error analysis by chemical subspace (reviewer deliverable 2)

`analysis/error_analysis.py` → `analysis/error_{base,log,combo_y2,frozen}.json`, `analysis/figs/error_*.png`.
**Out-of-fold only**: each polymer is scored by the members whose *test* split contained it (1120–1140 residuals
per arm, each polymer × property appearing in ~1.25 test splits), through `analysis/splits.py` so the frozen
runs' corrupted `test_idx.txt` is never trusted. Errors are log₁₀(predicted / measured).

| | frozen (K = 3) | `base` (K = 10) | `log` | **`combo_y2`** |
|---|---|---|---|---|
| median signed log ratio | +0.079 | +0.045 | +0.002 | **+0.010** |
| fraction over-predicted | 0.64 | 0.56 | 0.50 | **0.52** |
| median \|log ratio\| | 0.160 | 0.171 | 0.153 | **0.167** |
| R²(log) | 0.735 | 0.816 | 0.890 | **0.888** |
| Spearman ρ | 0.845 | 0.833 | 0.944 | **0.947** |
| non-positive (impossible) predictions | 11 / 342 | **58 / 1140** | **0** | **0** |

**Where the baseline fails, and what fixes it:**

| subspace | `base` R²(log) (ρ) | `log` | **`combo_y2`** |
|---|---|---|---|
| bottom H₂-permeability tertile | **−0.46** (0.41), +0.14 dex high | +0.59 (0.83) | **+0.67 (0.87)** |
| middle / top permeability tertile | 0.87 / 0.94 | 0.90 / 0.92 | 0.87 / 0.92 |
| fingerprint cluster 1 (the low-permeability cluster, n = 37) | **+0.06** (0.56), +0.15 dex high | +0.76 (0.87) | **+0.79 (0.90)** |
| fingerprint clusters 0 / 2 / 3 | 0.93 / 0.85 / 0.92 | 0.94 / 0.89 / 0.89 | 0.94 / 0.87 / 0.90 |
| family `polyimide (other)` | **+0.14** (0.74), +0.20 dex high | — | — |
| family `numbered entry` | 0.62 (0.68) | — | — |
| families PIM / PIM-PI / KAUST-PI | 0.93 / 0.92 / 0.86 | — | — |
| Murcko scaffold groups (4) | 0.67 … 0.86 | — | — |
| motif-count tertiles (3) | 0.74 … 0.85 | — | — |

**The baseline's error is systematic and it tracks permeability level, not scaffold or molecule size.** It is
good on the high-free-volume PIMs it has most of (R²(log) 0.93) and it systematically over-predicts the least
permeable polymers by ~0.15 dex with no useful rank ordering inside that group (ρ 0.41). Scaffold groups and
motif-count tertiles are comparatively uniform, so this is **not** a scaffold-generalization failure. The log
target repairs it almost completely. Worst individual polymers under `base`: PI-TB-5 0.63, PMDA-SBF 0.58,
PI-TB-2 0.53, 6FDA-HSBF-TR 0.52, TDA1-APAF 0.49, PI-TB-4 0.48 (median \|log ratio\| over their properties).

### 6.1 Cluster- and scaffold-held-out generalization — the sharpest answer to the reviewer

Hold out one whole group at a time (test = the entire group, 11–38 polymers; train on the rest); `fw` selected on
validation; mean over the four hold-outs and the 6 properties.

| stress test | mean test R² | median test R² | mean test ρ |
|---|---|---|---|
| held-out **fingerprint cluster**, raw targets (`holdout_fp`) | **−6.18** | −0.41 | +0.40 |
| held-out **fingerprint cluster**, log targets (`holdout_fp_log`) | −1.95 | −0.29 | +0.44 |
| held-out **Murcko scaffold group**, raw targets (`holdout_scaf`) | **+0.31** | +0.42 | +0.58 |
| held-out **Murcko scaffold group**, log targets (`holdout_scaf_log`) | −0.20 | +0.36 | +0.60 |

compared with mean test R² +0.36 … +0.62 and ρ 0.75–0.80 under random splits.

**The model generalizes across scaffolds but not across permeability regimes.** Holding out a scaffold group
costs relatively little (R² +0.31, ρ 0.58) because the scaffold groups were built property-mixed. Holding out a
fingerprint cluster is catastrophic, and the worst case is exactly cluster 1, the low-permeability cluster
(R² = −16.9 for H₂, −27.7 for O₂, −32.8 for CO₂): a model trained only on high-permeability chemistries cannot
predict a permeability regime it has never seen. Log targets reduce the cluster-hold-out damage threefold
(−6.18 → −1.95) but do not remove it, and they neither help nor hurt the scaffold hold-out in any way the data
can resolve (median R² +0.42 → +0.36, mean ρ +0.58 → +0.60; the mean R² difference, +0.31 → −0.20, comes
entirely from one of the four hold-outs and is not meaningful).

This is consistent with §6 and gives the reviewer a precise answer: the generalization limit is in the
**property dimension**, not the structural one, and random 80/20 splits overstate it because 44% of polymers
have a near neighbour (Tanimoto > 0.7) that stays in training.

## 7. Two bugs found in the frozen package (neither changes Table 2)

1. **`test_idx.txt` in `results_v2/outputs/pred/*` is corrupted.** Those runs were written before the snapshot's
   `main.py` fix, by `idx_partition(all_idx, all_idx, …)`, which indexes the shuffled permutation by itself, so
   the file holds `[all_idx[i] for i in all_idx[int(0.8N):]]`. Verified: for the runs checked it equals the
   doubly-indexed permutation exactly, and it shares 3–4 indices with `val_idx.txt`, which a real split cannot
   do. `val_idx.txt`, `mean_and_std.txt` and all `heldout_*` metrics are correct, so **Table 2, the SI ablation
   and the ensembles are unaffected**; only per-polymer attribution is. `PROVENANCE.md` §2 lists this as fixed,
   which is true of the snapshot's code but not of the delivered run dirs. `analysis/splits.py` detects it by
   the `val_idx.txt` overlap, recomputes the split from the seed, and asserts disjointness; campaign runs are
   clean (verified on `base` and `tseed`). Nothing under `results_v2/outputs/` was modified.
2. **The `fw = 1` (grammar-W feature) training path is not reproducible.** A byte-identical copy of the snapshot
   code reproduces `fw = 0` runs **exactly** (max absolute difference 0.000 across 399 epochs × 39 metric
   columns) but `fw = 1` runs differ by ~1e-7 at epoch 1 and, because training is chaotic (§4), by O(1) in test
   R² by epoch ~300. The frozen Table 2's selected O₂/N₂ and CO₂/CH₄ models are `fw = 1`. `PROVENANCE.md`'s
   "reruns differ by at most 0.1%" holds for predictions from a fixed checkpoint, not for the training
   trajectory.

## 8. What did not work, and what was still running

**Did not work:**

- **Smoothed validation selection (`--val_smooth 25`) hurts.** `log_smooth` V = 0.248 against `log` 0.164, and
  validation Spearman 0.730 against 0.785, so the pre-committed rule rejects it too. A zero-compute preview on
  the frozen grid had already shown no trend (V = 0.199 / 0.157 / 0.203 / 0.206 / 0.202 / 0.213 for windows
  1 / 5 / 10 / 25 / 50 / 100); the window was fixed at 25 a priori rather than chosen from that preview, which
  would have been test-set selection. Note `combo_y`, which contains the smoothing, is still second-best
  (V = 0.155) — the log target dominates and the smoothing only takes a little back.
- **Stratification on its own does nothing — this is the campaign's clearest negative result.** With raw
  targets: `strat_y` (permeability tertiles) V = 0.435, paired +0.063 ± 0.054 (t = 1.16); `strat_y2` (joint
  permeability × selectivity strata, the scheme that *does* help on top of the log target) V = 0.428, paired
  **+0.023 ± 0.075 (t = 0.30)**; `strat_scaf` (label-free Murcko-scaffold groups) V = 0.581, i.e. **worse than
  `base`**, paired −0.054 ± 0.104 (t = −0.52). `strat_y` also leaves the catastrophic splits in place (H₂/N₂
  still ± 0.74, CO₂ ± 0.64). So the §3 mechanism is real but equalising SST by stratification is not enough on
  its own: with raw targets and an MSE loss the *numerator* is still dominated by a handful of extreme
  polymers. The log target fixes the numerator, and only then does balancing the denominator add anything.
  Label-free stratification (scaffold groups, and `strat_struct` which was still running) is therefore not a
  usable fallback — it is the *property*-aware strata that matter.
- **Message-passing direction does not matter at this dataset's resolution.** Paired at K = 10 (n = 60):
  undirected − directed test R² = +0.058 ± 0.069 (t = 0.84), test MAE = −0.020 ± 0.019 (t = −1.08). Undirected
  is nominally better on both; neither is significant. The frozen ablation's ordering (directed MAE 0.456 vs
  undirected 0.468) **reverses** at K = 10 (directed 0.465 vs undirected 0.445) and sits well inside the noise.
  This agrees in sign with the group's ICML rebuttal (undirected 0.24 ± 0.08 vs directed 0.26 ± 0.11). It
  contradicts nothing in the *published* paper, whose edge-weight/random-walk claim was already retracted for
  v2; it does contradict the **revised** Methods sentence and SI Table S10, which should be softened.

- **Property-aware strata beat label-free strata even with the log target held fixed.** `combo_scaf` (log +
  smoothing + Murcko-scaffold strata) gives V = 0.180 and validation Spearman 0.735, against `combo_y2`'s
  V = 0.136 and 0.806. Together with `strat_scaf` being worse than the baseline on its own, the conclusion is
  that a label-free stratification is not a usable substitute: the strata have to be built on the property
  quantiles.

**Still running when this was written** (`STATUS.md` has the commands to fold them in): `strat_struct`
(motif-count strata, label-free, 40 of 60 started), `huber` (`--huber 1.0`, 40 of 60 started), and `smooth`
(`val_smooth 25` alone, queued). None can change the recommendation: the log target carries the entire
significant effect (paired t = 2.91 on its own, while every arm that changes only the split is t ≤ 1.16),
smoothing has already been shown to hurt, and `strat_struct` is a third label-free stratification of the kind
that has now failed twice. **`huber` is the one genuinely open question** — it attacks the same heavy-tail
problem as the log target but in the loss rather than the target, so if it works it would be an alternative
that keeps the raw target space and therefore avoids the §1/§9 "which space is R² in" caveat entirely. Worth
looking at when it lands.

**Not attempted:** fingerprint dimension / radius sweep (the plumbing is in place — `FPB=`/`FPR=` job tokens and
an env-driven `mol2fp`), black (core) vs red (R-group) feature cross, and pretrained-GNN initialization, which
`PLAN.md` §4 put out of scope on expected value per GPU-hour.

## 9. Secondary headline S — ensemble spread on the 1000 generated structures

`screening.py` was validated before use: given the 9 frozen selected members it reproduces the published counts
**372 / 3 / 378** and median O₂ ensemble std **251.7 barrer** exactly.

| | frozen (3 members) | `base` (10) | `log` (10) | **`combo_y2`** (10) |
|---|---|---|---|---|
| median O₂ ensemble std (barrer) | 251.7 | 327.6 | 122.6 | **83.2** |
| median O₂ **relative** std | 0.455 | 0.467 | 0.707 | 0.661 |
| median predicted O₂ (barrer) | 594 | 658 | 182 | **131** |
| H₂/N₂ count (std < 500, < 10) | 372 | 200 | 574 | 409 |
| **O₂/N₂ count (std < 100, < 2)** | **3** | **0** | 223 | **568** |
| CO₂/CH₄ count (std < 1000, < 5) | 378 | 35 | 575 | **616** |
| dataset polymers with negative predicted permeability | 2 | — | **0** | **0** |

**Two things must be said together here.** The absolute spread genuinely falls — median O₂ ensemble std 327.6 →
83.2 barrer, now below the paper's 100-barrer threshold — and the O₂ screening count goes from 3 (frozen) or 0
(`base` at matched K) to **568 of 1000**. But the **relative** ensemble std does not improve (O₂ 0.47 → 0.66),
because the log models predict much smaller permeabilities for the generated set (median O₂ 658 → 131 barrer).
So the gain on the generated set is a **calibration** gain, not a tighter-agreement gain. That the calibration
gain is real is supported independently of the generated set: out of fold the over-prediction bias falls from
+0.045 to +0.010 dex, all 58 impossible negative predictions disappear, and the low-permeability subspace is
repaired (§6). The cross-split *metric* improvement (§1) is a genuine agreement gain and is scale-free. Also
note that going from 3 to 10 ensemble members makes the published error bar **wider** (372/3/378 → 200/0/35 for
the frozen protocol), because a 3-member population std is biased low; the comparison that matters is
`combo_y2` against `base` at the same K = 10.
