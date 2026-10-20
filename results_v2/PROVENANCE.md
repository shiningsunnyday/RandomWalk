# Provenance of the revised results (v2)

Where every quantitative result in the revised main text and SI comes from, and how to regenerate it.
The submitted version's numbers are reproduced separately in `repro_v1/` (see `repro_v1/README.md`).
All paths below are relative to `results_v2/` unless they start with `~`.

Revised documents (written in place on 2026-10-19; the submitted versions are your backups):
- `~/RandomWalk/AI-driven Discovery and Experimental Validation of Polymer Structures for Gas Separation Membranes - Submission Candidate.docx`
- `~/RandomWalk/CSAIL_Paper_SI_V3.docx`

Both pass the docx schema validator and were rendered to PDF and checked page by page around every edit.

## 1. Frozen inputs

| Item | Location | Notes |
|---|---|---|
| Code | `code/` | snapshot of the repo at `HEAD dd797a8` plus the uncommitted fixes recorded in `CODE_VERSION` (git diff of `main.py`, `models/gnn_conv.py`, `utils/graph.py`, `utils/walk.py`; `code/scripts/` are the untracked repo scripts plus `GC_DIR`/`WALKS_FILE` overrides) |
| Dataset | `data/group-contrib/` | 114 corrected walks and properties (`polymer_walks_v2_preprocess.txt`); their clean SMILES (`polymer_walks_v2_smiles.txt`); 99 motif groups (the 97 canonical groups, 11 edited in place plus P36's R-site fix, and P42 and S33 appended); DAGs (`dags.pkl`) |
| Motif graph | `data/group-contrib/red_graph.edgelist` | rebuilt 2026-10-19 after the P36 fix (49,879 edges). Compared with the earlier build (50,549): every motif pair not involving P36 is edge-for-edge identical, and the set of connected motif pairs is unchanged. P36 is in none of the 114 training walks, so the grammar and the predictors did not need retraining |
| Grammar | `grammar/ckpt.pt` | `logs/logs-1792365901.742276` (trained 2026-10-18 on the corrected dataset with the paper recipe: `context_L`, α = 1e-3, Adam, 2000 epochs). A retrain with `TORCH_SEED=0` (`train_grammar.sh`) is byte-identical |
| Checksums | `MANIFEST.md5` | data, code and grammar checkpoint |

Python environment: `~/miniconda3/envs/polymer_walk/bin/python`, with `PYTHONPATH` set to the stub modules (`moses`, `eden`) used throughout. Retro* uses `~/miniconda3/envs/retro_star_env`. The docx edits use a py3.10 venv with lxml.

## 2. Fixes made for v2 (in the repo and the snapshot)

| Fix | File | Effect on results |
|---|---|---|
| Model selection on a validation split carved from the training split (`--val_size 0.2`). Checkpoints are kept at the lowest validation loss, and the test split is only evaluated (`heldout_*`) | `main.py` | v1 selected epochs, configurations and checkpoints on the test set |
| Normalization from training-split statistics only (`--train_norm`) | `main.py` | v1 normalized with all labelled polymers |
| Substituents must sit on R sites during generation: `edge_respects_r_sites` in `guess_edge_conn` and in `walk_enumerate_mols` | `utils/graph.py`, `utils/walk.py` | generated structures cannot bond at a non-R position |
| **Edge weights were never used.** `MyGINConv` passed `edge_weight` to `propagate` but inherited GIN's `message(x_j)`, which drops it, so `--edge_weights` did nothing in v1 or in the first v2 grid | `models/gnn_conv.py` | the 72 edge-weight runs were retrained after the fix. The invalid runs are kept in `outputs/pred_invalid_edgeweights_noop/` |
| `test_idx.txt` recorded the wrong polymers: `idx_partition(all_idx, all_idx, …)` indexed the shuffled list by itself. The split actually used for training and testing was always correct | `main.py` | bookkeeping only; v1's stored `test_idx.txt` files are also wrong |
| P36's R group 1 was annotated on a ring carbon instead of its methyl stand-in | `all_groups/extra/36.txt`, `all_extra*.txt`, `scripts/overhaul_groups_2026_10_15.py` | 30 of the first v2 best-first structures failed RDKit sanitization; the motif graph was rebuilt |
| Clean-SMILES builder: if dropping every unconsumed R stand-in breaks a ring, the R atoms that are ring members are kept (H-capped) | `scripts/regen_smiles_diag.py` | the 114 dataset SMILES are unchanged (audited); all 1000 best-first structures now sanitize |
| Seeds fixed everywhere (`TORCH_SEED=0`, `PYTHONHASHSEED=0`, Python/NumPy seeds) | run scripts | runs regenerate bit-for-bit on the same machine. Predictions re-run in another session differ by at most 0.1% (CPU float ordering) |

## 3. Protocol (decided before looking at test results)

- **Splits:** 3 split seeds. Each is an 80/20 train/test split, with 20% of the training split held out for validation (≈60 train / 15 validation / 19 test for H₂/N₂).
- **Grid:** per gas pair, edge weights {on, off} × motif-graph features W {on, off} × init {normal, zeros} × message passing {undirected, directed} × 3 splits, for 144 runs (`outputs/pred_grid_jobs.txt`; `train_predictor.sh`). Each run is 1000 epochs of a 5-layer GIN (hidden 16) with Adam, lr 1e-3.
- **Checkpoint:** the lowest-validation-loss checkpoint of each run.
- **Model selection:** per gas pair, the configuration with the lowest validation loss averaged over the three splits. That configuration's run on each split is the model for that split. Selected configurations: directed, no edge weights, normal init. W features are used for O₂/N₂ and CO₂/CH₄ but not for H₂/N₂.
- **Ensembles** (Fig 5, Tables S2/S4/S9, Figs S2–S4): the three selected models per gas pair, one per split. Error bars are the population std over them (`ensemble.py`).

## 4. Results → sources

| Result | Value(s) | Produced by | Raw source |
|---|---|---|---|
| **Table 1** Valid (Walk) / (SMARTS) / (SMILES) | 63% / 100% / 90% (v1: 56 / 100 / 91) | `compute_numbers.py` | `outputs/gen/random_n1000_s0_counters.json` from `gen.py random 1000 0` (1754 walks tried, 646 walk-invalid, 108 SMILES-invalid) |
| **Table 1** Unique / Novel / Diversity | 62% / 100% / 0.91 (v1: 96 / 100 / 0.90) | `compute_numbers.py` | `outputs/gen/bestfirst_n1000_clean.txt` (`gen.py bestfirst 1000` then `clean_smiles.py`). Uniqueness is now on clean canonical SMILES; v1 kept atom maps, which made identical molecules look distinct (see BRIEFING). `uniqueness.py` decomposes the 62%: all 1000 walks are distinct as walks and as motif graphs, 751 as closed walks up to rotation (a repeat unit is enumerated once per starting motif), 983 as atom-mapped SMILES, 625 as clean canonical SMILES. The sentences in the main text come from `outputs/gen/uniqueness.json` |
| **Table 1** Retro* | 54% (v1: 38%) | `~/RandomWalk/repro_v1/retro.py` (50 iterations, top-50 expansions, as in the paper) | `outputs/retro/bestfirst_clean_retro.txt` |
| **Table 2** MAE / R² (header MSE → MAE) | H₂ 0.47/0.48, H₂/N₂ 0.50/0.70, O₂ 0.35/0.51, O₂/N₂ 0.62/0.54, CO₂ 0.37/0.49, CO₂/CH₄ 0.43/0.75 (v1 R²: 0.69–0.90) | `summarize.py` → `compute_numbers.py` | `outputs/grid_runs.csv`, `outputs/table2.txt` (also lists MSE and the 9 selected run dirs) |
| Evaluation-protocol paragraph | — | `apply_main.py` | §3 above |
| **Fig 5** (PNG replaces the EMF) | — | `plots.py fig5()` | `outputs/fig5_ensemble.txt` (`predict.py --walks inputs/experimental_walks.txt` + `ensemble.py`); experimental values = SI Tables S3/S5 (unchanged) |
| Membrane-testing paragraph | H₂ overprediction 2.1–4.4×; CO₂/CH₄ selectivity within 25%; H₂/N₂ selectivity within 54% | `compute_numbers.py` | `fig5_ensemble.txt` vs SI Tables S3/S5 |
| Discussion: motif statistics | 99 motifs, 14.4 heavy atoms on average (v1: 97, 14.6); 2.0 motifs per generated monomer, at most 3 (v1: "<4") | `compute_numbers.py` | `data/group-contrib/all_groups/*.mol`, best-first walks |
| Methods: predictor paragraph and ablation sentence | directed MAE 0.456 vs undirected 0.468; edge weights 0.518/0.520 | `summarize.py` → `ablation.json` → `compute_numbers.py` | `outputs/grid_runs.csv` |
| **SI Tables S2 / S4** (predicted permeability / selectivity of the four PIM-PIs) | — | `apply_si.py` | `outputs/fig5_ensemble.txt` |
| **Literature vs in-house measurement** (530/150/1100 against 220/40/227 barrer, 2.4–4.8×) | — | `compute_numbers.py` | Our PIM-PI-4 is dataset row `PIM-PI-1`. Verified structurally, not by name: `L5->S20[->P17 ×4]->L5` and the dataset's `L16-0>S32-...-6>L16` build to the same C47H38N2O8 57-atom structure once S20's methyl attachment points are set to the dataset's (our bracket notation does not pin them down; of the three distinct structures the four methyls can give, one is the dataset row). That polymer is in **train** for split seed 1 (all pairs) and seed 3 (CO₂), in **validation** for seed 3 (H₂, O₂), and in **test** only for seed 2, so its Figure 5 prediction is partly a memorized literature label — see BRIEFING_2 §2.5 |
| **SI Tables S3 / S5** (experimental) | unchanged | — | — |
| **SI Table S6** (dataset walks and SMILES; 114 rows, SMILES column) | — | `apply_si.py` | `data/group-contrib/polymer_walks_v2_preprocess.txt`, `polymer_walks_v2_smiles.txt` |
| **SI Table S7** (dataset permeabilities) | unchanged | — | — |
| **SI Table S8** (1000 generated SMILES) | — | `apply_si.py` | `outputs/gen/bestfirst_n1000_clean.txt` |
| **SI Table S9** (their predicted properties) | — | `apply_si.py` | `outputs/s9_ensemble.txt` (`predict.py --walks outputs/screening_walks.txt` with the 9 selected runs, then `ensemble.py`) |
| **SI Table S10** (new: ablation) | — | `apply_si.py` | `outputs/ablation.json` |
| **SI Figs S2–S4** and caption counts | 372 / 3 / 378 structures within the error thresholds (v1: 29 / 33 / 82) | `plots.py screening_figures()` | `outputs/s9_ensemble.txt`, `outputs/dataset_ensemble.txt` (`predict.py --dataset`), `outputs/figs/screening_counts.json` |
| SI screening paragraphs (motif usage; "seeds" → "splits") | — | `motif_usage.py`, `apply_si.py` | `outputs/motif_usage.json`, `outputs/motif_usage_paragraph.txt` |
| SI Fig S1 caption (99 groups) | — | `apply_si.py` | — |
| **SI Tables S12 / S13** (new: prediction baselines, tuned and restricted to the raw target) | 15 methods each, under the §3 protocol; a cell given as ≪0 is an R² below −10 | `baselines/bl_summarize.py` → `apply_si.py` | `baselines/outputs/si_tables.json`, built from `baselines/outputs/results.jsonl` |
| **SI Table S14** (new: generation baselines, with the 114 dataset monomers as a reference row) | Retro* vs median heavy atoms: Pearson r = −0.80 (p = 0.0019), Spearman −0.54 (p = 0.068, not significant), n = 12 (all 13 rows, 12 with both numbers); dataset 9% at 52 heavy atoms vs ours 54% at 16. Both the caption and the main-text sentence are generated from `si_gen_table.json`, so they track the table as rows land | `baselines/bl_gen_si.py` → `apply_si.py` | `baselines/generation/metrics.jsonl`, `baselines/generation/retro/` (our own `repro_v1/retro.py`, 24 workers, two passes) |
| Generation-metrics paragraph: Retro*-vs-size caveat | same | `apply_main.py` | `baselines/outputs/si_gen_table.json` |
| **SI Table S11** (new: where sampling fails) | 98.5% of sampled trajectories never close; then 36.8% invalid walk, 9.7% unsanitizable | `compute_numbers.py` → `apply_si.py` | `outputs/gen/random_n1000_s0_counters.json` |
| Generation-metrics paragraph (Valid (Walk) is conditional on closure) | 1.5% of 116,822 trajectories close | `apply_main.py` | same |
| Membrane-testing paragraph: literature-vs-lab passage | durene PI, literature 530/150/1100 vs measured 220/40/227 barrer (2.4–4.8×) | `compute_numbers.py` → `apply_main.py` | dataset row `PIM-PI-1` of `polymer_walks_v2_preprocess.txt`; SI Tables S3/S5 via `plots.EXP` |
| Discussion: molecular-weight paragraph (new) | Mn 5.8–24.9 kDa, Đ 1.42–1.96, BET 391–479 m² g⁻¹; Spearman(Mn, permeability) = −0.40, p = 0.6, n = 4 | `compute_numbers.py` → `apply_main.py` | `inputs/gpc.json` (transcribed from the synthesis section of the submitted main text) |
| Discussion: "Future work." paragraph (new) | — | `apply_main.py` | — |
| **SI Table S1** (preliminary screening) | **not changed**: preliminary samples from an early run (`screening_v1`), not reproducible | — | — |
| Main Table 3 (experimental) | unchanged | — | — |

Logs of every applied text edit: `outputs/apply_main.log`, `outputs/apply_si.log`.

## 5. Regenerating everything

```bash
cd ~/RandomWalk
PY=~/miniconda3/envs/polymer_walk/bin/python; export PYTHONPATH=<stubs dir> TORCH_SEED=0 PYTHONHASHSEED=0
bash results_v2/train_grammar.sh                                   # optional; byte-identical to grammar/ckpt.pt
xargs -P 8 -L 1 bash results_v2/train_predictor.sh < results_v2/outputs/pred_grid_jobs.txt   # 144 runs
$PY results_v2/summarize.py                                        # table2.txt, grid_runs.csv, ablation.json
$PY results_v2/gen.py bestfirst 1000; $PY results_v2/gen.py random 1000 0      # ~1 h / ~20 min
$PY results_v2/clean_smiles.py results_v2/outputs/gen/bestfirst_n1000.json results_v2/outputs/gen/bestfirst_n1000_clean.txt
#   -> columns 1 and 3 of that file give bestfirst_n1000_clean_smiles.txt and outputs/screening_walks.txt
~/miniconda3/envs/retro_star_env/bin/python repro_v1/retro.py results_v2/outputs/gen/bestfirst_n1000_clean_smiles.txt results_v2/outputs/retro/bestfirst_clean_retro.txt 24
# the Unique decomposition, and the evidence that a deduped 1000 is not worth enumerating (BRIEFING_2 §3):
$PY results_v2/gen.py bestfirst 1400          # 8.6 h; its first 1000 reproduce bestfirst_n1000 exactly
$PY results_v2/clean_smiles.py results_v2/outputs/gen/bestfirst_n1400.json results_v2/outputs/gen/bestfirst_n1400_clean.txt
$PY results_v2/uniqueness.py                  # -> outputs/gen/uniqueness.json
# which of the four synthesized polymers are novel, and whether the known one's label was in training:
(cd results_v2/code && PYTHONPATH=$STUBS $PY ../exp_identity.py)
$PY results_v2/predict.py --walks results_v2/outputs/screening_walks.txt --out results_v2/outputs/s9_members.csv <9 selected run dirs from table2.txt>
$PY results_v2/ensemble.py results_v2/outputs/s9_members.csv results_v2/outputs/s9_ensemble.txt
#   (likewise --walks inputs/experimental_walks.txt -> fig5_*, and --dataset -> dataset_*)
$PY results_v2/plots.py; $PY results_v2/motif_usage.py; $PY results_v2/compute_numbers.py
bash results_v2/build_docx.sh [out_dir]      # unpack -> merge_runs -> apply_main/apply_si -> zip -> validate
#   Rebuilds both documents from results_v2/inputs/submitted/*.docx, which are the pristine submitted files
#   (main md5 9676904e423c29280af3f717f0bed1e7, SI md5 29d9371859c6130b990bbde2dde97756). Verified: before the
#   briefing-2 edits were added, this reproduced both revised documents' document.xml byte-for-byte. It does not
#   install the result; copy <out_dir>/out/{main,si}.docx over the working copies once the validator is clean.
```

## 5b. Predictor campaign (`campaign/`)

1434 runs over 27 arms, K = 10 split seeds, arm chosen on validation Spearman alone; deliverables in
`campaign/{PLAN,RESULTS,RECOMMENDATION,STATUS}.md`, per-run index in `campaign/outputs/campaign_runs.csv`.
Nothing in `campaign/` is used by the submitted documents except the paired directed-vs-undirected statistic in the
Methods ablation sentence, which `compute_numbers.py` recomputes from `campaign/outputs/campaign_runs.csv`.

Two bugs it surfaced in the frozen package, neither affecting any reported number:

| Bug | Effect |
|---|---|
| `outputs/pred/*/test_idx.txt` records the wrong polymers (pre-fix `idx_partition`) | metrics unaffected (the runs used the correct split); any per-polymer attribution built on those files is wrong. `campaign/analysis/splits.py` detects it and recomputes from the seed |
| the `fw = 1` training path is not bit-reproducible (~1e-7 at epoch 1 → O(1) in test R² by epoch ~300; `fw = 0` reproduces exactly) | the frozen selected O₂/N₂ and CO₂/CH₄ models are `fw = 1`, so those two rows cannot be reproduced bit-for-bit from the seed |

A third bug, in the campaign's own code copy and found on 2026-10-20, affects only the campaign: the `--huber` patch
in `campaign/code/main.py` assigned `loss_func` above an unconditional `loss_func = nn.MSELoss(...)` in the same
function, so **every `huber` arm reported before that date trained on plain MSELoss** and was numerically identical
to `base` (paired ΔR² = 0.000 over 60 runs at each of δ = 1.0, 0.5, 0.2). The patch was moved to the live
assignment, the invalid arms were deleted from `campaign/outputs/`, and the deltas were re-run. No other arm is
affected, because every other arm wanted MSELoss and received it, and nothing in the submitted documents is derived
from a `huber` arm.

## 5c. Motif-feature arms (`campaign/motif_emb.py`)

Six arms, 360 runs, testing pretrained per-motif embeddings against the 2048-bit Morgan fingerprint the predictor
ships with. Embeddings are written one `.npy` per motif, in `main.py`'s index order, and loaded with
`--mol_feat dir --mol_feat_dir <dir>`; `--mol_feat fp dir` concatenates. Two steps, because the `llamole` env has
`transformers` but not `sklearn`, which `utils.graph` imports:

```bash
cd results_v2/campaign
PYTHONPATH=$STUBS ~/miniconda3/envs/polymer_walk/bin/python motif_emb.py --smiles   # -> outputs/motif_emb/motif_smiles.txt
~/miniconda3/envs/llamole/bin/python motif_emb.py                                   # -> outputs/motif_emb/{polybert,chemberta}/<i>.npy
```

polyBERT is `xushijie/polyBERT` (600-d), ChemBERTa is `DeepChem/ChemBERTa-77M-MTR` (384-d), both mean-pooled over
the final hidden state. Atom map numbers are stripped before embedding so the comparison against `mol2fp` is
like-for-like. The `*_std` variants standardise per dimension and rescale to the fingerprints' measured mean L2
norm of 3.64, which removes a scale confound worth about a fifth of the gap. `onehot` is `--mol_feat one_hot`, a
99-way identity indicator with no chemistry, and is the diagnostic: it matches the fingerprint on MAE
(ΔMAE +0.028, t = 1.41 over 60 paired cells), so the node-feature channel is saturated by identity alone. All six
arms lose to `base`; none is used by the submitted documents.

## 5d. Literature-vs-lab calibration (`exp_calibrate.py`)

`~/miniconda3/envs/polymer_walk/bin/python exp_calibrate.py` → `outputs/calibration.{json,md}`. One factor per
property, `c = literature / our measurement`, computed on the reference polymer alone (our `PIM-PI-4`, which is
dataset row `PIM-PI-1`); calibrated prediction = prediction / c, evaluated on the three novel polymers
(`PIM-PI-1/2/3` in our numbering), which contribute nothing to `c`. "Within error" means
`|pred − obs| ≤ err + sd`, with `err` the measurement uncertainty and `sd` the ensemble spread, as elsewhere in the
paper. H₂ calibrates to a geometric-mean ratio of 1.00 (0/3 → 3/3 within error); O₂ and CO₂ overcorrect and the
selectivities get worse. **Nothing in the submitted documents is calibrated** — the factor rests on n = 1 and, for
O₂ and CO₂, on a polymer whose literature labels were in training.

## 5e. Unique-cycle generation (`gen_unique.py`)

`PYTHONPATH=$STUBS ~/miniconda3/envs/polymer_walk/bin/python gen_unique.py <n> [seed]` →
`outputs/gen/unique_n<n>.{json,_trajs.pkl,_stats.json}`. Best-first enumeration as in `code/`, with
`diffusion.verify_walk` wrapped so that a closed walk whose canonical cycle (canonical over rotation and
reflection, as `uniqueness.py` computes it) has already been accepted raises and is treated as an invalid walk.
A cycle is marked seen only once it is known to assemble, so an invalid rotation cannot block a valid one. The
wrapper is removed before the records are built, so every stored record comes from the unmodified path, and an
assertion checks that the accepted walks have pairwise-distinct cycles. `code/` is not modified. Verified exact at
n = 250: the output is the original `bestfirst` enumeration with duplicate cycles dropped and order preserved
(element-by-element, 21 skipped).

## 6. Prediction baselines (`baselines/`)

Added after the first revision, for the reviewers' request for competitive comparisons. Every baseline runs under
the protocol in §3: the same three splits (recomputed from the seed in `bl_data.py`, **not** read from the frozen
`outputs/pred/*/test_idx.txt`, which record the wrong polymers), validation-only hyperparameter selection,
training-split normalization, and MAE/R² on the normalized targets. Each baseline is additionally offered the
log-permeability target as a hyperparameter chosen on validation, with a `[raw target]` control that removes it.

| File | Role |
|---|---|
| `bl_data.py` | dataset, per-gas-pair masks (N = 94/94/92), splits, featurizers, metrics, result-file locking |
| `bl_classical.py` | GC regression, Bag-of-Motifs ±expert, Morgan+RF/GP/SVR, RDKit descriptors +XGBoost/ridge, polyBERT and ChemBERTa + heads, training-mean reference |
| `bl_polybert_embed.py` | polyBERT (`xushijie/polyBERT` mirror) and ChemBERTa-77M-MTR embeddings of the 114 monomers |
| `bl_chemprop.py` | Chemprop D-MPNN, given our splits as `--separate_val_path` / `--separate_test_path` |
| `hmgnn/` | HM-GNN and HM-GNN (+expert) port (see `HMGNN.md`) |
| `bl_summarize.py` | `outputs/results.jsonl` + our own grid → `outputs/comparison.{md,json}` |
| `bl_k10.py` | the same methods at ten split seeds against the frozen three → `outputs/k10.{md,json}` |
| `generation/` | generation baselines for Table 1 (see its own `RESULTS.md`) |

```bash
PY=~/miniconda3/envs/polymer_walk/bin/python
$PY baselines/bl_classical.py                     # all feature-based baselines
$PY baselines/bl_classical.py --raw_only          # the raw-target control
~/miniconda3/envs/llamole/bin/python baselines/bl_polybert_embed.py
~/miniconda3/envs/reinvent4/bin/python baselines/bl_chemprop.py --epochs 60 --gpu 0
~/miniconda3/envs/reinvent4/bin/python baselines/bl_chemprop.py --epochs 60 --gpu 0 --raw_only
$PY baselines/bl_summarize.py
```

Superseded outputs are kept for reference: `outputs/gen_oldgraph/` (samples from the pre-P36-fix graph) and `outputs/pred_invalid_edgeweights_noop/`.

**The same baselines at ten split seeds** (`outputs/results_k10.jsonl`, `outputs/k10.{md,json}`), for decision 0 of
BRIEFING_2. Identical protocol with `SEEDS = 1..10`, the seed set the campaign's K=10 arms used, so the two are
directly comparable; our own rows come from the campaign `base`, `log` and `combo_y2` arms. Nothing in
`results.jsonl` or the SI tables is touched by it.

```bash
$PY results_v2/baselines/bl_classical.py --seeds 1,2,3,4,5,6,7,8,9,10 --out results_v2/baselines/outputs/results_k10.jsonl
~/miniconda3/envs/reinvent4/bin/python results_v2/baselines/bl_chemprop.py --epochs 60 --gpu 0 --seeds 1,2,3,4,5,6,7,8,9,10 \
    --out results_v2/baselines/outputs/results_k10.jsonl
$PY results_v2/baselines/bl_k10.py      # -> outputs/k10.md
```

Thirteen of the fifteen methods have both protocols: the twelve classical ones and Chemprop. HM-GNN has **not** been
re-run at ten seeds (it is 0.524 MAE at three against our 0.456, so the comparison is not close), and neither have
the `[raw target]` controls, which exist only to isolate the log-target option at the frozen protocol.
