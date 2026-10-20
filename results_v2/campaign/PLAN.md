# Campaign plan: reduce cross-split disagreement of the property predictor

Written 2026-10-19 20:10 EDT, **before** any new-configuration run was launched.
(The only job launched before this file existed is the *control* arm, which is the frozen v2 protocol itself,
re-run unchanged at 10 split seeds instead of 3 — see §4, arm `base`.)

Everything is written under `results_v2/campaign/`. Nothing under `results_v2/outputs/`, `grammar/`, `data/`
or the `.docx` files is touched. Shared repo code (`~/RandomWalk/main.py`, `diffusion.py`, `models/`, `utils/`)
is not touched; the campaign uses a copy of the frozen snapshot at `campaign/code/` and all new behaviour is
behind flags that default to the frozen behaviour (§3).

---

## 1. Headline metric (fixed before running)

**Primary — V: cross-split disagreement of test metrics.**

> V = mean over the 6 properties (H2, H2/N2, O2, O2/N2, CO2, CO2/CH4) of the across-split population
> standard deviation of test R², where each property's test R² is computed in the space the model was
> trained in (exactly the quantity Table 2 reports), at the validation-selected checkpoint of the
> validation-selected configuration, over **K = 10 split seeds for every arm**.

Lower is better. Frozen v2 baseline at K = 3: V = mean(0.27, 0.08, 0.36, 0.13, 0.34, 0.01) = **0.198**.
The control arm re-estimates V at K = 10 and that is the number every other arm is compared against,
because a population std over 3 samples is biased low and noisy; comparing a 3-split std to a 10-split
std would be rigged. **Every arm is run at K = 10, 1000 epochs, same seeds 1..10.**

**Primary — S: spread of predictions on the 1000 generated structures.**

> S = median over the 1000 best-first structures of the O2 ensemble standard deviation (barrer), and
> the SI Figs S2–S4 screening counts (H2/N2, O2/N2, CO2/CH4 structures passing the paper's thresholds).

Frozen v2: median O2 ensemble std ≈ 250 barrer; counts 372 / 3 / 378. The O2 count (3 of 1000) is the
blocking number.

**Accuracy guards, reported next to V and S for every arm so a variance win that costs accuracy is visible:**

- mean test R² (same space as V) — must not drop materially.
- mean test MAE on the normalized scale (Table 2's MAE).
- **mean test Spearman ρ** — added because it is invariant to the target transform and is therefore the only
  accuracy number that compares a raw-target model with a log-target model fairly.
- mean test R² **in raw barrer space** for every arm (log-target predictions are inverted with 10^·), for the
  same reason. Where a raw-target model predicts a negative permeability, log-space R² is undefined and is
  reported as such rather than silently dropped.

**Honesty note carried into RESULTS.md and RECOMMENDATION.md:** R² in log space and R² in raw space are not
the same quantity. If a log-target arm wins, Table 2 has to say which space it reports, and both are given.

## 2. Protocol invariants (non-negotiable, unchanged)

- 80/20 train/test per split seed; 20% of the training split held out as validation.
- The **validation split alone** selects the epoch, the configuration and the checkpoint (`--val_size 0.2`).
- The test split is only ever evaluated (`heldout_*` columns).
- Normalization from training-split statistics only (`--train_norm`).
- `TORCH_SEED=0 PYTHONHASHSEED=0`, every command recorded (`campaign/jobs/*.txt` + `campaign/train.sh`).
- No arm, configuration, epoch or checkpoint is ever chosen by a test number. Within an arm, the per-pair
  configuration (`fw` on/off) is chosen by mean validation loss over the 10 splits, as `summarize.py` does.
- **Choosing between arms** is itself a decision rule and must not use test numbers. The rule, fixed here:
  the arm is chosen by **mean validation Spearman ρ over the 10 splits and 6 properties** (transform-invariant,
  so it compares across target spaces; validation only). V and S are *reported* for every arm, and the chosen
  arm's V and S are the result. If the validation-chosen arm is not the arm with the best V, that is stated.

  > **Amendment, 2026-10-19 21:15 EDT** — made before any arm had finished, so before any arm's V was known.
  > The two control arms `base` and `base_undir` were launched at 20:08 and 20:10, a few minutes before the
  > metrics patch landed, so their `metrics.csv` files carry no Spearman columns, and re-running them is not
  > worth 120 runs of compute. The arm-selection rule is therefore **mean validation R² in raw barrer space**
  > over the split seeds and the 6 properties. It is available for every arm (for a raw-target run the plain
  > validation R² *is* the raw-space one, verified to machine precision in the smoke test), it is
  > validation-only, and it is comparable across target spaces, which is the property the rule needed.
  > Validation Spearman is still reported for every arm that has it, and the paired statistical tests against
  > `base` use test R², which every arm has.

  > **The amendment above is REVERTED, 2026-10-19 22:55 EDT. The rule is the original one: mean validation
  > Spearman ρ.** Two things changed. First, the amendment was unnecessary: `analysis/val_spearman.py` recovers
  > the missing validation Spearman for the control arms **exactly**, without retraining, by loading each run's
  > own validation-selected checkpoint and predicting its own validation polymers — cross-checked against the
  > `*_spearman` columns of the 120 `log`-arm runs, max |difference| 0.00e+00. Second, and more importantly,
  > the amended rule was wrong on the merits: **validation R² in raw barrer space is, up to the affine
  > normalization, the training objective of a raw-target run and not of a log-target run**, so it
  > systematically favours the raw-target arms. That is a property of the objective, visible without looking at
  > any test number, and it is the reason for reverting. It does change the outcome, which is exactly why this
  > is recorded rather than quietly fixed: under the amended rule the chosen arm would have been `base_undir`
  > (validation raw R² 0.669 vs `log` 0.490); under the original pre-committed rule it is `log`
  > (validation Spearman 0.785 vs `base_undir` 0.673 and `base` 0.650). Both columns are reported for every arm
  > in `outputs/headline.txt`, so a reader can apply either rule. No test number entered this decision.

## 3. New flags added to `campaign/code/main.py` (all default to frozen behaviour)

| Flag | Default | Effect |
|---|---|---|
| `--log_target` | off | regress log10 of both targets (permeability and selectivity) instead of raw barrer / raw ratio; normalization statistics are then log-space training statistics |
| `--stratify {none,y,struct}` | `none` | stratified split construction (§4) instead of a plain seeded shuffle |
| `--huber DELTA` | `0` (= MSE) | smooth-L1 training loss with the given delta |
| `--weight_decay W` | `0` | Adam weight decay |
| `--val_smooth K` | `1` (= current argmin) | epoch/checkpoint selection on a centred running mean of the validation loss over K epochs |
| extra reporting | — | `heldout_*_spearman`, `heldout_raw_*_{r^2,mae,spearman}`, and the validation counterparts, added as new `metrics.csv` columns |

Backward compatibility is checked by re-running split seeds 1, 2, 3 of the frozen configuration with the
campaign copy and confirming the frozen `heldout_*` numbers reproduce (arm `base`, seeds 1–3 vs
`results_v2/outputs/grid_runs.csv`).

## 4. Arms, in order, with the reason and the budget

One run = 1 gas pair, 1 split seed, 1000 epochs, ≈ 25–55 min on one core. 128 cores available, ~60 runs in
parallel. One arm = 3 pairs × `fw` ∈ {0,1} × 10 seeds = **60 runs ≈ 1 wave ≈ 1 h**. `ew = 0`, `init = normal`,
directed message passing are fixed at the frozen validation-selected values throughout and are not re-searched.

| # | Arm | What changes | Why I expect it to help | Cost |
|---|---|---|---|---|
| 0 | `base` | nothing — frozen protocol at K = 10 | control; also re-estimates V honestly | 60 runs (launched 20:08) |
| 1 | `log` | `--log_target` | permeabilities span >3 orders of magnitude. With raw targets, MSE is dominated by the few most permeable polymers, so each split's loss surface (and its training std, used for normalization) is set by which outliers landed in it — a direct mechanism for split-to-split disagreement. It also explains the median best epoch of 68 and the 25th percentile of **8** in the frozen grid: validation loss is minimised almost immediately, i.e. the selected checkpoint is nearly untrained on many splits. Log targets also make negative predicted permeability impossible (BRIEFING §5) and should compress the ensemble spread on the generated set | 60 runs |
| 2 | `smooth` | `--val_smooth 25` | the validation split is ~15 polymers, so the validation-loss curve is noisy and its raw argmin is a lottery; a 25-epoch running mean is still a validation-only rule but a far less noisy one. Expected to cut V on its own | 60 runs |
| 3 | `strat_y` | `--stratify y` | stratify the split by tertile of log permeability so every test split spans the property range. Removes the largest single source of test-R² variance (a test split that happens to be narrow has a tiny target variance, so R² collapses) | 60 runs |
| 4 | `strat_struct` | `--stratify struct` | label-free stratification (tertile of walk length / motif count). Weaker but does **not** use y to form the split, so it has no leakage caveat at all | 60 runs |
| 5 | best-of-1–4 combined | the winning combination | the mechanisms are independent | 60 runs |
| 6 | `hparam` | on top of 5: `--huber 1.0`, `--weight_decay 1e-4`, `--dropout_rate 0.1`, `--hidden_dim 32` as four separate one-at-a-time changes | regularization with 114 samples; Huber is the loss-level version of the outlier argument | up to 4 × 60 runs, screened at 3 pairs × 10 seeds for the 2 most promising only |
| 7 | fingerprint size/radius; black/red (core vs R-group) feature cross | `mol2fp` dimension and radius; separate core and R-group descriptors with an interaction | only if 1–6 leave V above ~0.12, and time remains. Highest implementation cost, least certain | 60–120 runs |
| — | pretrained-GNN init | — | **explicitly out of scope**: most expensive per unit of expected variance reduction | 0 |

Then, for the chosen arm:

8. **Ensemble and screening re-run** (no training): predict the 1000 generated structures and the 114 dataset
   polymers with the 10 selected checkpoints, recompute S and the SI Figs S2–S4 counts with
   `plots.py`-equivalent logic, and produce the new Table 2. ~1 h of CPU.
9. `RESULTS.md`, `RECOMMENDATION.md` written; `STATUS.md` kept current throughout.

## 5. Compute budget

- Available: 128 cores, ~60 concurrent runs, overnight ≈ 10–12 h ⇒ ~10 waves ⇒ ~600 runs.
- Allocated: arms 0–4 (5 waves, ~5 h), arm 5 (1 wave), arm 6 (2 waves), ensemble/screening (1 wave-equivalent),
  slack 1–2 waves. Arm 7 only if slack survives.
- Stop rule: if no arm improves V by more than its own uncertainty, say so plainly in RESULTS.md and
  recommend keeping the frozen protocol with K = 10 splits (which at least estimates the error bar honestly).

## 6. Skepticism

With ~19 test polymers per split, a single split's R² has a standard error of roughly ±0.1–0.2. V itself,
estimated from 10 splits, has a relative uncertainty of about 1/sqrt(2·9) ≈ 24%. **A reduction in V of less
than ~25% relative is not a result.** I will also report, for each arm, the paired per-(pair, seed) difference
in test R² against the control on identical splits, which is far more powerful than comparing two means, and
a bootstrap over splits for V.
