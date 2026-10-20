# RECOMMENDATION

Written 2026-10-20 03:05 EDT. Supported by **ten completed random-split arms of 60 runs at K = 10, a 72-run
variance-replication arm and four 24-run hold-out arms: 768 runs, every one 1000 epochs.** Two lower-priority
arms were still running (`strat_struct`, `huber`, with `smooth` queued); `RESULTS.md` §8 lists them and says why
none can change this. Full evidence: `RESULTS.md`. Protocol and
decision rules: `PLAN.md`. Live state: `STATUS.md`.

---

## 1. Adopt three changes

1. **Regress log₁₀ of both targets** (`--log_target`): log₁₀(permeability) and log₁₀(selectivity) instead of raw
   barrer and raw ratio. This carries essentially the whole effect.
2. **Stratify the split on joint permeability × selectivity strata** (`--stratify balanced --strata_file
   analysis/strata_y2_<pair>.txt`): three tertiles of log permeability × two halves of log selectivity, applied
   proportionally to the train, validation **and** test blocks. This adds a small but statistically solid gain
   on top of the log target.
3. **Use 10 split seeds, not 3.** This is not optional and is independent of the other two.

Everything else is unchanged: same 5-layer GIN, hidden 16, Adam lr 1e-3, 1000 epochs, `ew = 0`, `init = normal`,
directed message passing, `fw` selected per gas pair on validation loss, `--train_norm`, 80/20 train/test with
20% of train held out for validation, validation-only selection of epoch, configuration and checkpoint, and the
test split only ever evaluated. This is campaign arm **`combo_y2`**.

### Why

| | frozen (K = 3) | `base` = frozen protocol at K = 10 | `log` | **`combo_y2`** |
|---|---|---|---|---|
| **V** = mean across-split std of test R² | 0.198 | 0.516 [0.225, 0.662] | 0.164 [0.114, 0.180] | **0.136 [0.079, 0.156]** |
| mean test R² | 0.580 | 0.363 | 0.602 | **0.619** |
| mean test MAE (normalized) | 0.456 | 0.465 | 0.465 | 0.477 |
| mean test Spearman ρ | — | — | 0.750 | **0.799** |
| worst per-property across-split std | 0.36 | **1.07** | 0.25 | **0.29** |
| out-of-fold median log₁₀(pred/measured) | +0.079 | +0.045 | +0.002 | **+0.010** |
| out-of-fold R²(log) / ρ | 0.735 / 0.845 | 0.816 / 0.833 | 0.890 / 0.944 | **0.888 / 0.947** |
| impossible (non-positive) predictions | 11 / 342 | **58 / 1140** | 0 | **0** |
| median O₂ ensemble std, 1000 generated | 251.7 | 327.6 | 122.6 | **83.2** |
| screening counts H₂/N₂ · O₂/N₂ · CO₂/CH₄ | 372 · **3** · 378 | 200 · **0** · 35 | 574 · 223 · 575 | 409 · **568** · 616 |
| **validation** Spearman ρ (the selection rule) | — | 0.650 | 0.785 | **0.806** |

Paired on identical splits, n = 60: `combo_y2` − `base` test R² = **+0.256 ± 0.076 (t = 3.35)**;
`combo_y2` − `log` test Spearman = **+0.048 ± 0.014 (t = 3.53)**. The completed factorial is unambiguous about
which ingredient matters: **every arm that changes only the split is indistinguishable from the baseline**
(paired t = +1.16 permeability-tertile strata, +0.30 joint strata, −0.52 scaffold strata, +0.84 undirected),
while every arm containing the log target clears t = 2.8. Stratification is worth having only on top of the
log target.

`combo_y2` is also what the **pre-committed** selection rule picks (highest mean validation Spearman). No test
number entered the choice. `PLAN.md` records an amendment to that rule and its reversal; the reversal is argued
from the objective — raw-space validation R² is the training objective of a raw-target run, so it is not
transform-neutral — and both columns are reported for every arm so a reader can apply either rule.

## 2. Exact commands to reproduce

```bash
cd ~/RandomWalk/results_v2/campaign
PY=~/miniconda3/envs/polymer_walk/bin/python
export PYTHONPATH=/tmp/claude-1026/-home-msun415-RandomWalk/e4c142f8-9c54-42f0-bb59-09e87958531b/scratchpad/stubs
export TORCH_SEED=0 PYTHONHASHSEED=0

# 0. code: a copy of the frozen results_v2/code/ plus backward-compatible flags (idempotent, already applied)
$PY patch_main.py

# 1. the stratum label files (also produces the dataset-bias figures and JSON)
$PY analysis/dataset_bias.py

# 2. 60 runs = 3 gas pairs x feat_concat_W {0,1} x 10 split seeds, 1000 epochs each
#    (~1 h at 60-way parallelism on 128 cores; jobs/combo_y2.txt is the exact job list, one line per run)
bash run_arm.sh jobs/combo_y2.txt combo_y2 60

# 3. selection (validation only), the headline metric and Table 2
$PY summarize_campaign.py combo_y2        # -> outputs/headline.txt, selected_combo_y2.csv, campaign_runs.csv
$PY analysis/val_spearman.py combo_y2     # -> outputs/val_spearman.csv  (the selection-rule column)
$PY make_table2.py combo_y2               # -> outputs/table2_combo_y2.txt (log space AND raw space)

# 4. the 10-member ensemble: generated-set spread, screening counts, dataset residuals
$PY screening.py combo_y2                 # -> outputs/screen_combo_y2_{ensemble,dataset,residuals,...}
$PY analysis/error_analysis.py combo_y2   # -> analysis/error_combo_y2.json, analysis/figs/error_combo_y2.png
```

One run on its own:

```bash
bash train.sh "0 1" 7 0 0 normal 1 1000 combo_y2 \
  --log_target --stratify balanced \
  --strata_file $PWD/analysis/strata_y2_01.txt
```

Property-column codes: `"0 1"` = H₂ and H₂/N₂, `"2 1"` = O₂ and O₂/N₂, `"4 3"` = CO₂ and CO₂/CH₄; the strata file
suffix matches (`strata_y2_01`, `_21`, `_43`). The selected members, one per split seed, are listed at the bottom
of `outputs/table2_combo_y2.txt`.

Nothing under `results_v2/outputs/`, `grammar/`, `data/` or the `.docx` files was written. Shared repo code
(`~/RandomWalk/main.py`, `diffusion.py`, `models/`, `utils/`) was not touched: all changes are in
`campaign/code/`, applied by `patch_main.py`, and every new flag defaults to the frozen behaviour — verified by
reproducing frozen `fw = 0` runs bit-for-bit (max absolute difference 0.000 over 399 epochs × 39 columns).

## 3. The new Table 2 this produces

Full file: `outputs/table2_combo_y2.txt`. K = 10 split seeds, mean ± population std across splits, test split
only ever evaluated.

**In log₁₀ space — the space the model is trained and selected in, and what Table 2 should report:**

| property | MAE | MSE | R² | Spearman ρ |
|---|---|---|---|---|
| H₂ | 0.42 ± 0.08 | 0.27 ± 0.11 | **0.73 ± 0.12** | 0.85 ± 0.06 |
| H₂/N₂ | 0.49 ± 0.06 | 0.37 ± 0.08 | **0.64 ± 0.08** | 0.80 ± 0.07 |
| O₂ | 0.41 ± 0.09 | 0.28 ± 0.12 | **0.68 ± 0.17** | 0.81 ± 0.08 |
| O₂/N₂ | 0.57 ± 0.06 | 0.53 ± 0.11 | **0.38 ± 0.29** | 0.69 ± 0.11 |
| CO₂ | 0.38 ± 0.06 | 0.25 ± 0.07 | **0.79 ± 0.06** | 0.85 ± 0.06 |
| CO₂/CH₄ | 0.60 ± 0.12 | 0.77 ± 0.38 | **0.50 ± 0.11** | 0.79 ± 0.07 |

Against the frozen Table 2 (K = 3, raw space: H₂ 0.48 ± 0.27, H₂/N₂ 0.70 ± 0.08, O₂ 0.51 ± 0.36,
O₂/N₂ 0.54 ± 0.13, CO₂ 0.49 ± 0.34, CO₂/CH₄ 0.75 ± 0.01): the means are better for H₂, O₂ and CO₂, slightly
worse for the three selectivities, and the error bars are comparable or smaller — while the honest comparison,
against the same protocol at K = 10, is an improvement on every property.

**Add the Spearman ρ column.** It is invariant to the target transform, so it makes the old and new tables
comparable, it is the quantity a screening tool is judged on, and it is the campaign's most robust result
(out-of-fold ρ 0.947 against 0.833 for the baseline).

O₂/N₂ remains the weakest property (R² 0.38 ± 0.29) and is the one place the validation-selected epoch is still
sometimes almost untrained (epochs 4, 9, 11 on three of the ten splits). If anything else is worth one more
wave of compute, it is O₂/N₂ specifically.

## 4. The new screening counts (SI Figs S2–S4)

`screening.py` reproduces the published 372 / 3 / 378 and median O₂ std 251.7 barrer exactly from the 9 frozen
members, so the pipeline is validated before being used on anything new.

| | frozen (3 members) | `base`, 10 members | **`combo_y2`, 10 members** |
|---|---|---|---|
| H₂/N₂ (std < 500 barrer, < 10) | 372 | 200 | **409** |
| O₂/N₂ (std < 100 barrer, < 2) | **3** | **0** | **568** |
| CO₂/CH₄ (std < 1000 barrer, < 5) | 378 | 35 | **616** |
| median O₂ ensemble std | 251.7 | 327.6 | **83.2** |
| dataset polymers with negative predicted H₂ | 2 | — | **0** |

The paper's O₂ threshold goes from keeping **3 of 1000** structures to keeping **568 of 1000**, and no generated
structure or dataset polymer receives a negative predicted permeability any more.

## 5. What it costs — read this before quoting anything above

1. **R² changes meaning, and in raw barrer space this does not help.** The headline R² gains are in log₁₀ space.
   In raw barrer space the same models are no better and are noisier: paired `combo_y2` − `base` raw-space test
   R² = −0.150 ± 0.147 (t = −1.02), and per property the raw-space numbers are H₂ −0.29 ± 1.95,
   H₂/N₂ 0.50 ± 0.16, O₂ 0.19 ± 0.54, O₂/N₂ 0.41 ± 0.26, CO₂ −0.06 ± 1.04, CO₂/CH₄ 0.54 ± 0.34. Both spaces are
   printed in `outputs/table2_combo_y2.txt`. The honest framing: a log target buys a large, significant
   improvement in log-space accuracy, in rank accuracy and in cross-split agreement, and buys nothing in
   absolute barrer accuracy for the most permeable polymers. Permeabilities span 3–4 decades and the use case is
   ranking and screening, so log space is the right space — but the paper must say which space it reports, and
   the Spearman column is what makes the claim transform-free.
2. **The stratified split changes what the test set means.** The test split is no longer a uniform random sample
   of the 114 polymers; it is balanced across three log-permeability tertiles × two log-selectivity halves. Test
   metrics are therefore estimates of performance on a **property-balanced** population, not on the dataset's own
   strongly skewed distribution. Stated plainly in the Methods, this is defensible and arguably more informative
   — the dataset's skew is an artefact of what has been published, not of the design space — but it is a real
   change and a reader must not read the new error bars as estimates for a random draw from the dataset. It also
   uses the labels of all 114 polymers to form the split (standard practice for stratified regression splits,
   and no test *prediction* is involved). **If that caveat is unacceptable, adopt `log` alone**: V = 0.164 instead
   of 0.136, screening counts 574 / 223 / 575 instead of 409 / 568 / 616, with a plain random split and no change
   to what the test set means. That keeps ~85% of the variance win and all of the calibration win.
   There is **no label-free substitute**: stratifying on Murcko-scaffold groups is worse than the baseline
   (V = 0.581 against 0.516, paired −0.054 ± 0.104), stratifying on the joint property strata without the log
   target does nothing at all (V = 0.428, paired +0.023 ± 0.075, t = 0.30), and scaffold strata *with* the log
   target (`combo_scaf`) reach only V = 0.180 and validation Spearman 0.735 against `combo_y2`'s 0.136 and
   0.806. It is specifically the property-quantile strata combined with the log target that work.
3. **The generated-set improvement is calibration, not tighter agreement.** The *relative* ensemble std on the
   1000 generated structures does not improve (O₂ 0.47 → 0.66) because the log models predict much smaller
   permeabilities (median O₂ 658 → 131 barrer). The paper's thresholds are absolute, so the counts improve a
   great deal. **Do not write "the ensemble agrees more closely on the generated set."** Write that the
   predictions are no longer inflated — which is supported independently of the generated set: out of fold the
   over-prediction bias falls from +0.045 to +0.010 dex, the 58 impossible negative predictions disappear, and
   the low-permeability subspace is repaired. If a reviewer asks for a *relative* error bar the screening gain
   evaporates, so state the absolute thresholds and the calibration fix together rather than claiming tighter
   agreement.
4. **Going from 3 to 10 splits makes the published error bars bigger, not smaller.** The frozen caption's
   372 / 3 / 378 is partly an artefact of a 3-member population std, which is biased low; re-running the *frozen*
   protocol at 10 members gives 200 / 0 / 35. Part of the improvement to 409 / 568 / 616 is undoing a loss that
   honest error bars create. The comparison that matters is `combo_y2` against `base` at the same K = 10.
5. **Compute:** 60 runs per grid at these settings instead of 36, about 1 h at 60-way parallelism on 128 cores.
   The ensemble has 10 members per gas pair instead of 3, so screening 1000 structures takes ~45 min instead of
   ~15.
6. **No test-set selection anywhere.** Epoch, configuration, checkpoint and the arm itself were all chosen on
   validation.

## 6. Secondary recommendations

1. **Average each split's metrics over a few torch seeds, or ensemble torch seeds into each split's member.**
   The `tseed` arm shows **43–45% of the across-split std budget is pure optimization noise at a fixed split**
   (within-split std over torch seeds 0.118–0.125 in test R², against an across-split std of 0.151–0.156). With
   4 torch seeds the combined std falls from 0.196 to 0.167 (≈15%) for 4× the compute, and it also makes the
   numbers reproducible, which matters given §6.3 below. This is a legitimate variance reduction — it reduces
   real model variance rather than redefining the metric — and it is independent of everything in §1.
2. **Soften the directedness claim in the revised Methods and SI Table S10.** Paired at K = 10 (n = 60):
   undirected − directed test R² = +0.058 ± 0.069 (t = 0.84), test MAE = −0.020 ± 0.019 (t = −1.08). Undirected
   is nominally better on both and neither is significant. The frozen ablation's ordering (directed MAE 0.456 vs
   undirected 0.468) **reverses** at K = 10 (directed 0.465 vs undirected 0.445), well inside the noise. This
   agrees in sign with the group's ICML rebuttal (undirected 0.24 ± 0.08 vs directed 0.26 ± 0.11) and
   contradicts nothing in the *published* paper, whose v1 edge-weight/random-walk claim was already retracted
   for v2. The supportable sentence is: "message-passing direction makes no difference we can resolve on 114
   polymers; edge weights hurt." SI Table S10 should carry K = 10 numbers and say n = 10 splits.
3. **Add the dataset-bias characterization and the subspace error analysis to the SI** — they answer Reviewer 1
   directly and are already computed (`analysis/dataset_bias.json`, `analysis/error_*.json`,
   `analysis/figs/bias_*.png`, `analysis/figs/error_*.png`; narrative in `RESULTS.md` §5–6). The three strongest
   points: (a) the dataset is *not* uniformly high-free-volume — four fingerprint clusters with median H₂
   permeability 132 / 545 / 1295 / 1368 barrer, families from 134 to 1625 barrer, none more than 29% of the data
   — yet five polymers of ~95 carry 63–71% of each property column's variance; (b) the baseline fails
   specifically on the **least** permeable polymers (bottom permeability tertile out-of-fold R²(log) **−0.46**,
   ρ 0.41, over-predicting by 0.14 dex), which the recommended configuration repairs (**+0.67**, ρ 0.87);
   (c) **held-out-cluster generalization fails** — holding out a whole fingerprint cluster gives mean test
   R² = −6.2 with raw targets and −2.0 with log targets, while holding out a whole Murcko scaffold group gives
   **+0.31**. The model generalizes across scaffolds but not across permeability regimes. That is the honest,
   specific answer to "generalizability to other classes is unclear", and it supports reframing the predictor as
   a within-domain ranking tool for microporous polymers built from this motif vocabulary.
4. **Fix the two bookkeeping problems in the delivered package** (`RESULTS.md` §7): `test_idx.txt` in
   `results_v2/outputs/pred/*` is corrupted (doubly indexed — Table 2, the ablation and the ensembles are
   unaffected, only per-polymer attribution), and the `fw = 1` grammar-W training path is not bit-reproducible,
   which affects the reproducibility claim for the selected O₂/N₂ and CO₂/CH₄ models but not their validity.
