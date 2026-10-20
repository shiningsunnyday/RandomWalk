# STATUS — generation baselines (live handoff)

Last updated **2026-10-20 06:52 EDT**. Paths relative to
`~/RandomWalk/results_v2/baselines/generation/`. Read `PLAN.md` for the design, `RESULTS.md` for the
table and the caveats.

## Short version

**All 11 baselines + the dataset reference row are final in `metrics.jsonl`.** Retro\* is measured
on 10 of the 11. Everything is done; nothing of mine is still running.

Retro\* passes: **run 1** (`retro/run1/` + `manifest_run1.json`, 5 rows x 45), **run 2**
(`retro/run2/` + `manifest_run2.json`, 4 rows x 25), and a JT-VAE pass (`manifest_run4.json`,
n=25, **20%**, verified SMILES-by-SMILES against `retro/combined_out.txt`, 25/25 agreement).

### ⚠ Read this before running anything here: a second session shares this directory

A **second, concurrent Claude Code session** (pid 536480, started 02:57, cwd marker
`/tmp/claude-ca38-cwd`; mine was `/tmp/claude-d262-cwd`) has been doing overlapping work in this
same directory. At **06:42:10** it ran its own `retro_tools.py build --k 25 --baselines jtvae`,
which **overwrote the shared scratch files** `retro/combined_in.txt`, `retro/combined_out.txt`,
`retro/combined.log` and `retro/manifest.json` while my own catch-up pass was mid-flight. It also
created `retro/run3/` and `manifest_run4.json`, and then ran `bl_gen_si.py` in `baselines/`.

Consequences, all already handled in `metrics.jsonl`:
* **JT-VAE 20% is correct and verified** (its pass was self-consistent; I re-derived it from
  `manifest_run4.json` + `retro/combined_out.txt` with 25/25 SMILES agreement).
* **DEG's Retro\* was invalidated and removed.** My pass had interleaved jtvae+deg into 50
  molecules; the collect at 06:42:33 read the *replaced* 25-entry jtvae-only manifest, so it paired
  manifest line indices against a differently-ordered output. The 4% that this produced was
  discarded as unverifiable (see `retro_note` on the deg row), and the raw files were overwritten
  before they could be re-paired.
* `retro/combined_in.txt`, `combined_out.txt`, `combined.log` and `manifest.json` are **shared
  scratch and are not a record of anything** — trust only `retro/run1/`, `retro/run2/` and the
  `manifest_run*.json` snapshots.

**To add DEG's Retro\* (~3 min on the now-idle machine), and to avoid a repeat collision, work on
private filenames or check `pgrep -af retro.py` first:**

```bash
G=~/RandomWalk/results_v2/baselines/generation; P=~/miniconda3/envs/polymer_walk/bin/python
pgrep -af "repro_v1/retro.py"      # make sure nobody else is mid-pass
$P $G/common/retro_tools.py build --k 25 --baselines deg && $G/retro/run3.sh
```

## Finished rows in `metrics.jsonl`

| row | Valid | Unique | Novel | Div | Retro\* (n) | median HA |
|---|---|---|---|---|---|---|
| dataset_114 (reference) | 100% | 97% | 0% | 0.82 | 8.9% (45) | 52 |
| hiervae | 100% | 22% | 54% | 0.85 | 16.0% (25) | 45 |
| graph_ga_uncond | 100% | 91% | 100% | 0.91 | 33.3% (45) | 26 |
| graph_ga_co2 | 100% | 100% | 100% | 0.60 | 0.0% (45) | 60 |
| smiles_lstm (early-stop) | 17% | 7% | 22% | 0.81 | 12.0% (25) | 39 |
| smiles_lstm_fullfit | 94% | 11% | 1% | 0.81 | 6.7% (45) | 50 |
| smiles_transformer (early-stop) | 0.2% | 0.1% | 100% | 0.00 | — | 1 |
| smiles_transformer_fullfit | 86% | 11% | 0.4% | 0.81 | 16.0% (25) | 54 |
| deg (grammar induction, no RL) | 100% | 50% | 60% | 0.81 | — (invalidated, see ⚠) | 43 |
| reinvent_tl | 88% | 42% | 53% | 0.85 | 13.3% (45) | 44 |
| reinvent_rl | 98% | 72% | 99% | 0.86 | 24.0% (25) | 23 |
| jtvae | 100% | 22% | 92% | 0.72 | 20.0% (25) | 57 |

## Still running

**Nothing of mine.** The other session's processes may still be active — check
`pgrep -af "repro_v1/retro.py"` and `pgrep -af bl_gen_si` before you start anything.

To redo any piece by hand:

```bash
G=~/RandomWalk/results_v2/baselines/generation; P=~/miniconda3/envs/polymer_walk/bin/python
$P $G/common/metrics.py <name> $G/runs/<name>/samples.txt   # recompute one row
$P $G/common/retro_tools.py build --k 25 --baselines <rows> # then $G/retro/run3.sh (edit paths)
$P $G/common/retro_tools.py collect && $P $G/common/make_table.py
```

## Findings that matter for the paper (details in RESULTS.md)

* **Retro\* rewards small molecules made of conventional building blocks.** Over the 11 rows with a
  Retro\* number: Pearson **r = −0.83**, Spearman **−0.63** against median heavy atoms. Size is a
  strong driver but not deterministic — JT-VAE has the largest molecules (57 HA) and still scores
  20%, above the **real dataset monomers (52 HA, 9%)**, because JT-VAE assembles from a 33-entry
  vocabulary of common motifs while the real PIMs/polyimides are large *and* chemically unusual.
  Our row is smallest *and* motif-assembled, so it wins on both axes; report the dataset row beside
  ours and make the size-matched comparison (REINVENT-RL 23 HA/24%, Graph GA 26 HA/33%) explicitly.
* Hier-VAE(+expert) is **not expressible**: only **11 of 96** distinct expert motifs ever appear as a
  `find_fragments()` output (25% of fragment occurrences). Dropped, documented, not faked.
* A character LM on 114 SMILES has two useless regimes: early-stopped → 17%/0.2% valid;
  fit to convergence → 94%/86% valid but **1% / 0.4% novel** (it memorises the training set).
* Graph GA unconditional matches us on Valid/Unique/Novel/Diversity and is the row to argue against
  on design-space grounds, not on these metrics.
* REINVENT RL shrinks molecules (median 44 → 23 HA) chasing the CO2 surrogate; it is the most
  size-comparable learned baseline to ours.
* DEG's grammar sampler works well (170 production rules, 1000 molecules with zero failures, 100%
  valid / 50% unique / 60% novel); it is the **RL loop** that was unaffordable. The reduced-budget
  DEG RL run was abandoned (noted in `runs/deg/train.log`) after failing to finish its first grammar
  evaluation (10 Retro\* calls) in ~1.5 h.
* JT-VAE's decoder is the slowest thing in this study: ~100 molecules/hour, which is why its row is
  the last to land. Its training was fine (53 epochs, loss 166 → 2.32).

## Conventions / gotchas for whoever picks this up

* conda pythons by absolute path. `polymer_walk` (torch 2.0.1, rdkit 2023.03) for hgraph2graph,
  metrics, the LMs, the GA; `torch_molecule` for JT-VAE; `DEG` for DEG; `reinvent4` for REINVENT;
  `retro_star_env` for Retro\*. **`mhg` has torch 1.1.0, which has no kernels for this A6000 (sm_86).**
* every job runs `nice -n 10`, `OMP_NUM_THREADS=1`. The other campaign kept load at ~100–165, so each
  of my processes got 25–60% of a core; that, not the GPU, set every timing in this run.
* nothing outside this directory was written. `common/pylibs/` holds a locally `pip --target`-ed
  matplotlib so no existing conda env was modified. `common/gbga/` symlinks (not copies) the two
  GB-GA operator files from `~/torch-molecule`.
* `runs/jtvae_smoke/` and `runs/_probe/` are throwaway probes (2-epoch JT-VAE smoke test; 24-molecule
  Retro\* timing probe), kept because the probe result (0/24 on real monomers) is cited in RESULTS.md.
