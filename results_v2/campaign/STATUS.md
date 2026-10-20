# STATUS — handoff

Last updated: 2026-10-20 03:10 EDT.

**The campaign has a conclusion.** Read `RECOMMENDATION.md` first, then `RESULTS.md`. `PLAN.md` holds the
pre-committed metric and decision rules plus two dated amendments (the second reverts the first). This file is
only about what is running, what is done and what to do next.

## Bottom line

Adopt **`combo_y2`**: log10 targets + a split stratified on joint permeability x selectivity strata + 10 split
seeds. V (mean across-split std of test R^2) falls **0.516 -> 0.136** against the identical protocol at K = 10,
paired test R^2 +0.256 +- 0.076 (t = 3.35, n = 60), and the O2 screening threshold goes from keeping 3 of 1000
generated structures to keeping 568. It is also what the pre-committed validation-only rule selects
(validation Spearman 0.806, the highest of any arm). The log target carries nearly all of the effect; a
log-only variant (`log`, V = 0.164) is the fallback if the stratified test-set definition is unwanted.

Two caveats that must travel with those numbers (both in `RECOMMENDATION.md` section 5): the R^2 gains are in
log space and do **not** hold in raw barrer space, and the generated-set improvement is a calibration gain,
not tighter relative agreement.

## Environment to resume anything here

```bash
cd ~/RandomWalk/results_v2/campaign
PY=~/miniconda3/envs/polymer_walk/bin/python
export PYTHONPATH=/tmp/claude-1026/-home-msun415-RandomWalk/e4c142f8-9c54-42f0-bb59-09e87958531b/scratchpad/stubs
export TORCH_SEED=0 PYTHONHASHSEED=0
```

## Still running

Expected to finish by roughly 04:30-05:30 EDT. `summarize_campaign.py` rescans every arm and now prints
incomplete arms, the hold-out arms and `tseed` in a separate block so they are never ranked against the
random-split arms by mistake.

| arm | runs | purpose | state at 03:10 |
|---|---|---|---|
| `strat_struct` | 60 | motif-count-tertile strata (label-free) | 40 of 60 started |
| `huber` | 60 | `--huber 1.0`, raw targets | 40 of 60 started |
| `smooth` | 60 | `val_smooth 25` alone, raw targets | queued |

**None of these can change the recommendation.** The completed factorial already shows that every arm changing
only the split is indistinguishable from the baseline (paired t = +1.16 permeability strata, +0.30 joint strata,
-0.52 scaffold strata), that smoothing hurts (`log_smooth` V = 0.248 against `log` 0.164), and that label-free
strata lose to property strata even with the log target (`combo_scaf` V = 0.180, validation Spearman 0.735,
against `combo_y2`'s 0.136 and 0.806). `strat_struct` is a third label-free stratification of the kind that has
now failed twice. `huber` is the one genuinely open question left - it attacks the same heavy-tail problem as
the log target but in the loss rather than the target.

Watch progress and fold them in:

```bash
for a in $(ls outputs | grep -v '\.'); do
  echo "$a $(ls outputs/$a/*/predictor_*/metrics.csv 2>/dev/null | wc -l) runs"
done
pgrep -fc main.py                                  # concurrent training processes; 128 cores available
tail -2 logs/queue_*.log

# when an arm reaches 1000 epochs on every run:
$PY analysis/val_spearman.py <arm>                 # the selection-rule column (exact, from the checkpoints)
$PY summarize_campaign.py                          # rescans ALL arms -> outputs/headline.txt + headline.json
$PY make_table2.py <arm>
$PY screening.py <arm>                             # only worth it for an arm that beats combo_y2's valrho 0.806
$PY analysis/error_analysis.py <arm>
```

If any arm's **validation** Spearman exceeds 0.806 it becomes the winner under the pre-committed rule, and
`RECOMMENDATION.md` sections 1-4 need re-pointing at it (`screening.py` and `error_analysis.py` give the
section 4 and section 6 numbers). Otherwise only `RESULTS.md` sections 1 and 8 need the extra rows.

## Done

- **768 completed runs across 15 arms** (ten 60-run random-split arms, `tseed` at 72, four 24-run hold-out
  arms), every one 1000 epochs, all seeded and reproducible. Exact job lists in
  `jobs/*.txt` (one line = one run); the drivers are `train.sh` / `run_arm.sh` / `queue.sh`.
- The split factorial is complete: log target alone (t = 2.91), joint strata alone (t = 0.30), permeability
  strata alone (t = 1.16), scaffold strata alone (t = -0.52), log + joint strata (t = 3.35 vs baseline, and
  +0.048 +- 0.014 in test Spearman over log alone). The log target is the whole effect.
- `campaign/code/` = a copy of the frozen `results_v2/code/` plus `patch_main.py` (idempotent). New flags:
  `--log_target`, `--strata_file`, `--stratify {none,balanced,holdout}`, `--holdout_stratum`, `--huber`,
  `--weight_decay`, `--val_smooth`, plus new `metrics.csv` columns (`val_loss_smooth`, `*_spearman`, `raw_*`).
  `utils/graph.py:mol2fp` takes `FP_BITS` / `FP_RADIUS` from the environment. **Every default is the frozen
  behaviour**, verified by reproducing frozen `fw = 0` runs bit-for-bit (max absolute difference 0.000 over
  399 epochs x 39 metric columns).
- Stratification verified: `balanced` is proportional in all three of train / validation / test; `holdout` puts
  exactly the held-out stratum in the test split.
- `screening.py` validated against the frozen deliverable: 372 / 3 / 378 counts and median O2 std 251.7 barrer
  reproduced exactly from the 9 frozen members.
- Dataset-bias characterization (`analysis/dataset_bias.*`, `analysis/figs/bias_*.png`) and out-of-fold
  subspace error analysis (`analysis/error_*.json`, `analysis/figs/error_*.png`) - the two reviewer-demanded
  deliverables - complete for `frozen`, `base`, `log` and `combo_y2`.
- All four cluster/scaffold hold-out stress tests complete (`holdout_fp`, `holdout_fp_log`, `holdout_scaf`,
  `holdout_scaf_log`): the model generalises across scaffolds (mean test R^2 +0.31, rho 0.58) but not across
  permeability regimes (mean test R^2 -6.18 raw, -1.95 with log targets).
- Variance decomposition complete (`tseed`): 43-45% of the across-split std budget is optimization noise.

## Not attempted

- Fingerprint dimension / radius sweep. The plumbing is in place and untested: add `FPB=512 FPR=3` as job-line
  tokens (`run_arm.sh` exports them and appends `_fp512r3` to the run tag so nothing collides).
- Black (core) vs red (R-group) feature cross. Would need a real change to `mol2fp` / `featurize_walk` plus
  either a bilinear layer or a count-sketch of the outer product; not started.
- Pretrained-GNN initialization - out of scope per `PLAN.md` section 4 (worst expected value per GPU-hour).
- O2/N2 is the weakest property left (test R^2 0.38 +- 0.29, and on three of ten splits the validation-selected
  epoch is still < 12, i.e. almost untrained). If there is budget for one more wave, spend it here.

## Notes and incidental findings

- **`test_idx.txt` in the frozen `results_v2/outputs/pred/*` dirs is corrupted. Do not read it.** Those runs
  were written before the snapshot's `main.py` fix, by `idx_partition(all_idx, all_idx, ...)`, which indexes
  the shuffled permutation by itself; the file holds `[all_idx[i] for i in all_idx[int(0.8N):]]`. Verified: for
  the runs checked it equals the doubly-indexed permutation exactly and shares 3-4 indices with `val_idx.txt`,
  which a real split cannot do. `val_idx.txt`, `mean_and_std.txt` and every `heldout_*` metric are correct, so
  Table 2, the SI ablation and the ensembles are unaffected - only per-polymer attribution is.
  `PROVENANCE.md` section 2 lists this as fixed, which is true of the snapshot's code but not of the delivered
  run dirs. `analysis/splits.py:test_positions` is the safe accessor (detects the bug via the `val_idx.txt`
  overlap, recomputes the split from the seed, always asserts disjointness); both analysis scripts use it and
  record which source they used. Campaign runs write correct indices. **An earlier version of `RESULTS.md`
  section 3 drew a conclusion from the corrupted file; it is withdrawn and corrected there.**
- Positions in those files index the **gas pair's labelled subset** in walks_file line order (94 rows for the
  H2 and O2 pairs, 92 for CO2/CH4), not the 114 walks_file lines.
- **The `fw = 1` (grammar-W) training path is not bit-reproducible**: `fw = 0` reproduces exactly, `fw = 1`
  diverges from ~1e-7 at epoch 1 to O(1) in test R^2 by epoch ~300. The frozen Table 2's selected O2/N2 and
  CO2/CH4 models are `fw = 1`.
- `outputs/1000/` was removed: 60 empty tag dirs from a first, failed runner invocation at 20:06 containing no
  metrics.
- Nothing under `results_v2/outputs/`, `grammar/`, `data/` or the `.docx` files was written by this campaign at
  any point, and no shared repo code outside `campaign/` was modified (`~/RandomWalk/main.py`,
  `utils/graph.py`, `models/gnn_conv.py`, `utils/walk.py` all carry mtimes from before this session began, and
  their modified state is the pre-existing one recorded in `CODE_VERSION`).
- **Another workstream was active in `results_v2/` during the night** and is not this campaign: new
  `baselines/` (`bl_classical.py`, `bl_chemprop.py`, `bl_polybert_embed.py`, `bl_data.py`, `bl_gen_si.py`),
  `BRIEFING_2.md`, `build_docx.sh`, and re-runs that touched the mtimes of `outputs/table2.txt`,
  `outputs/grid_runs.csv` and `outputs/numbers.json`. I checked that the frozen reference numbers this campaign
  quotes are unaffected: `outputs/table2.txt` still reads H2 0.48 +- 0.27, H2/N2 0.70 +- 0.08, O2 0.51 +- 0.36,
  O2/N2 0.54 +- 0.13, CO2 0.49 +- 0.34, CO2/CH4 0.75 +- 0.01, exactly the values used throughout
  `RESULTS.md` and `RECOMMENDATION.md`. Worth coordinating before anyone regenerates `outputs/` again.
