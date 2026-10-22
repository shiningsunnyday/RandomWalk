# Prediction baselines under the v2 protocol

Same three train/test splits, same validation-only model selection, same training-split normalization as
Table 2. MAE and R² are on the normalized targets, mean ± population standard deviation over the three
splits. `log` marks how many of the 18 (property, split) fits selected the log-target option on validation
(our own model is not offered that option, so its cell is blank); a `[raw target]` row is the same baseline
with that option removed.

Rows are ranked by mean MAE, not mean R². A model fitted on log-permeability can extrapolate above the
training maximum on a single high-permeability test polymer, and exponentiating that error produces an R²
of −30 on that one split while MAE hardly changes; the mean R² of such a row is meaningless. `median R²`
and `worst R²` are over all 18 (property, split) fits and make those rows visible.

## Averaged over the six properties

| Method | mean MAE ↓ | mean R² ↑ | median R² | worst R² | log |
|---|---|---|---|---|---|
| Bag-of-Motifs (+expert) | 0.393 | 0.632 | 0.678 | -0.52 | 16/18 |
| polyBERT + ridge | 0.401 | 0.661 | 0.703 | -0.03 | 10/18 |
| Morgan FP + GP (Tanimoto) | 0.417 | 0.625 | 0.559 | 0.40 | 6/18 |
| Ours (connection-aware) | 0.421 | 0.625 | 0.610 | 0.15 |  |
| Morgan FP + GP (Tanimoto) [raw target] | 0.428 | 0.620 | 0.567 | 0.40 | 0/18 |
| RDKit descriptors + XGBoost | 0.435 | 0.552 | 0.540 | -0.04 | 12/18 |
| polyBERT + GP (RBF) | 0.436 | 0.635 | 0.626 | 0.44 | 4/18 |
| Bag-of-Motifs (+expert) [raw target] | 0.439 | 0.526 | 0.590 | -0.97 | 0/18 |
| Morgan FP + random forest | 0.444 | 0.606 | 0.590 | 0.40 | 0/18 |
| Morgan FP + random forest [raw target] | 0.444 | 0.606 | 0.590 | 0.40 | 0/18 |
| Morgan FP + SVR (RBF) | 0.445 | 0.467 | 0.579 | -0.84 | 8/18 |
| polyBERT + GP (RBF) [raw target] | 0.447 | 0.611 | 0.618 | 0.33 | 0/18 |
| Morgan FP + SVR (RBF) [raw target] | 0.449 | 0.569 | 0.579 | 0.15 | 0/18 |
| Ours (random-walk grammar GNN) | 0.456 | 0.580 | 0.630 | 0.06 |  |
| polyBERT + ridge [raw target] | 0.468 | 0.641 | 0.623 | 0.40 | 0/18 |
| RDKit descriptors + XGBoost [raw target] | 0.485 | 0.462 | 0.554 | -0.67 | 0/18 |
| Chemprop D-MPNN | 0.487 | 0.559 | 0.544 | 0.33 | 8/18 |
| Chemprop D-MPNN [raw target] | 0.512 | 0.570 | 0.593 | 0.33 | 0/18 |
| ChemBERTa + ridge ⚠ | 0.518 | 0.366 | 0.531 | -1.01 | 18/18 |
| Bag-of-Motifs | 0.521 | 0.450 | 0.398 | 0.17 | 4/18 |
| HM-GNN [raw target] | 0.524 | 0.438 | 0.527 | -0.04 | 0/18 |
| HM-GNN (+expert) [raw target] ⚠ | 0.526 | -0.391 | 0.370 | -10.83 | 0/18 |
| Bag-of-Motifs [raw target] | 0.534 | 0.439 | 0.413 | 0.17 | 0/18 |
| GC regression (motif counts) | 0.541 | 0.423 | 0.514 | -0.36 | 8/18 |
| ChemBERTa + ridge [raw target] | 0.561 | 0.445 | 0.474 | 0.03 | 0/18 |
| HM-GNN ⚠ | 0.577 | -2.476 | 0.324 | -39.36 | 18/18 |
| GC regression (motif counts) [raw target] | 0.593 | 0.414 | 0.504 | -0.36 | 0/18 |
| HM-GNN (+expert) ⚠ | 0.593 | -6.422 | 0.609 | -100.82 | 18/18 |
| Training mean (reference) | 0.785 | -0.065 | -0.044 | -0.22 | 0/18 |
| Training mean (reference) [raw target] | 0.785 | -0.065 | -0.044 | -0.22 | 0/18 |
| RDKit descriptors + ridge [raw target] ⚠ | 2.732 | -281.054 | 0.357 | -3145.25 | 0/18 |
| RDKit descriptors + ridge ⚠ | 4.142 | -604.575 | 0.268 | -5820.22 | 10/18 |

⚠ = at least one (property, split) fit with R² < −1, i.e. a degenerate extrapolation; read MAE for these rows.

## MAE ↓ per property

| Method | H2 | H2/N2 | O2 | O2/N2 | CO2 | CO2/CH4 |
|---|---|---|---|---|---|---|
| Ours (connection-aware) | 0.34 ± 0.17 | 0.47 ± 0.27 | 0.39 ± 0.17 | 0.61 ± 0.20 | 0.39 ± 0.21 | 0.32 ± 0.09 |
| Ours (random-walk grammar GNN) | 0.47 ± 0.21 | 0.50 ± 0.28 | 0.35 ± 0.19 | 0.62 ± 0.17 | 0.37 ± 0.18 | 0.43 ± 0.13 |
| Bag-of-Motifs | 0.49 ± 0.22 | 0.55 ± 0.29 | 0.56 ± 0.17 | 0.61 ± 0.04 | 0.50 ± 0.26 | 0.44 ± 0.10 |
| Bag-of-Motifs (+expert) | 0.30 ± 0.09 | 0.45 ± 0.27 | 0.26 ± 0.12 | 0.52 ± 0.07 | 0.32 ± 0.17 | 0.51 ± 0.12 |
| Bag-of-Motifs (+expert) [raw target] | 0.36 ± 0.13 | 0.50 ± 0.32 | 0.37 ± 0.11 | 0.54 ± 0.03 | 0.32 ± 0.18 | 0.54 ± 0.14 |
| Bag-of-Motifs [raw target] | 0.57 ± 0.16 | 0.54 ± 0.22 | 0.56 ± 0.17 | 0.61 ± 0.04 | 0.50 ± 0.26 | 0.44 ± 0.10 |
| ChemBERTa + ridge | 0.56 ± 0.27 | 0.45 ± 0.19 | 0.57 ± 0.39 | 0.60 ± 0.14 | 0.47 ± 0.27 | 0.45 ± 0.16 |
| ChemBERTa + ridge [raw target] | 0.63 ± 0.20 | 0.50 ± 0.20 | 0.59 ± 0.22 | 0.64 ± 0.14 | 0.57 ± 0.23 | 0.43 ± 0.07 |
| Chemprop D-MPNN | 0.45 ± 0.24 | 0.54 ± 0.21 | 0.40 ± 0.27 | 0.64 ± 0.11 | 0.50 ± 0.20 | 0.41 ± 0.10 |
| Chemprop D-MPNN [raw target] | 0.49 ± 0.19 | 0.55 ± 0.21 | 0.46 ± 0.23 | 0.64 ± 0.09 | 0.53 ± 0.16 | 0.39 ± 0.08 |
| GC regression (motif counts) | 0.43 ± 0.22 | 0.46 ± 0.18 | 0.52 ± 0.28 | 0.74 ± 0.21 | 0.55 ± 0.31 | 0.55 ± 0.16 |
| GC regression (motif counts) [raw target] | 0.58 ± 0.13 | 0.63 ± 0.32 | 0.53 ± 0.27 | 0.70 ± 0.25 | 0.56 ± 0.30 | 0.56 ± 0.15 |
| HM-GNN | 0.72 ± 0.29 | 0.28 ± 0.05 | 0.53 ± 0.28 | 0.54 ± 0.10 | 1.01 ± 0.39 | 0.39 ± 0.10 |
| HM-GNN (+expert) | 0.33 ± 0.05 | 0.63 ± 0.20 | 0.98 ± 0.76 | 0.61 ± 0.20 | 0.63 ± 0.36 | 0.38 ± 0.12 |
| HM-GNN (+expert) [raw target] | 0.48 ± 0.10 | 0.58 ± 0.28 | 0.38 ± 0.13 | 0.63 ± 0.16 | 0.61 ± 0.23 | 0.47 ± 0.09 |
| HM-GNN [raw target] | 0.50 ± 0.22 | 0.58 ± 0.18 | 0.49 ± 0.25 | 0.57 ± 0.12 | 0.59 ± 0.34 | 0.42 ± 0.15 |
| Morgan FP + GP (Tanimoto) | 0.36 ± 0.16 | 0.47 ± 0.23 | 0.34 ± 0.18 | 0.56 ± 0.12 | 0.35 ± 0.18 | 0.42 ± 0.07 |
| Morgan FP + GP (Tanimoto) [raw target] | 0.39 ± 0.13 | 0.50 ± 0.21 | 0.36 ± 0.15 | 0.56 ± 0.11 | 0.36 ± 0.17 | 0.40 ± 0.09 |
| Morgan FP + SVR (RBF) | 0.38 ± 0.09 | 0.48 ± 0.21 | 0.36 ± 0.15 | 0.58 ± 0.10 | 0.41 ± 0.12 | 0.45 ± 0.11 |
| Morgan FP + SVR (RBF) [raw target] | 0.39 ± 0.14 | 0.51 ± 0.20 | 0.38 ± 0.13 | 0.58 ± 0.10 | 0.38 ± 0.15 | 0.45 ± 0.11 |
| Morgan FP + random forest | 0.40 ± 0.14 | 0.50 ± 0.23 | 0.39 ± 0.15 | 0.59 ± 0.13 | 0.37 ± 0.17 | 0.42 ± 0.12 |
| Morgan FP + random forest [raw target] | 0.40 ± 0.14 | 0.50 ± 0.23 | 0.39 ± 0.15 | 0.59 ± 0.13 | 0.37 ± 0.17 | 0.42 ± 0.12 |
| RDKit descriptors + XGBoost | 0.46 ± 0.23 | 0.50 ± 0.31 | 0.34 ± 0.16 | 0.59 ± 0.15 | 0.39 ± 0.24 | 0.33 ± 0.13 |
| RDKit descriptors + XGBoost [raw target] | 0.46 ± 0.23 | 0.59 ± 0.31 | 0.43 ± 0.18 | 0.57 ± 0.16 | 0.49 ± 0.30 | 0.38 ± 0.18 |
| RDKit descriptors + ridge | 3.23 ± 3.82 | 3.21 ± 4.06 | 1.68 ± 1.52 | 6.96 ± 9.03 | 9.26 ± 12.41 | 0.51 ± 0.10 |
| RDKit descriptors + ridge [raw target] | 3.25 ± 3.80 | 3.32 ± 3.99 | 1.65 ± 1.54 | 7.02 ± 8.99 | 0.56 ± 0.21 | 0.59 ± 0.14 |
| Training mean (reference) | 0.71 ± 0.07 | 0.92 ± 0.26 | 0.66 ± 0.11 | 0.95 ± 0.07 | 0.67 ± 0.12 | 0.81 ± 0.17 |
| Training mean (reference) [raw target] | 0.71 ± 0.07 | 0.92 ± 0.26 | 0.66 ± 0.11 | 0.95 ± 0.07 | 0.67 ± 0.12 | 0.81 ± 0.17 |
| polyBERT + GP (RBF) | 0.41 ± 0.13 | 0.48 ± 0.21 | 0.38 ± 0.14 | 0.56 ± 0.18 | 0.40 ± 0.17 | 0.39 ± 0.10 |
| polyBERT + GP (RBF) [raw target] | 0.44 ± 0.08 | 0.48 ± 0.21 | 0.42 ± 0.11 | 0.55 ± 0.18 | 0.40 ± 0.17 | 0.39 ± 0.10 |
| polyBERT + ridge | 0.35 ± 0.02 | 0.47 ± 0.24 | 0.30 ± 0.07 | 0.61 ± 0.17 | 0.34 ± 0.07 | 0.33 ± 0.11 |
| polyBERT + ridge [raw target] | 0.42 ± 0.07 | 0.50 ± 0.21 | 0.42 ± 0.12 | 0.62 ± 0.16 | 0.45 ± 0.21 | 0.40 ± 0.07 |

## R² ↑ per property

| Method | H2 | H2/N2 | O2 | O2/N2 | CO2 | CO2/CH4 |
|---|---|---|---|---|---|---|
| Ours (connection-aware) | 0.64 ± 0.20 | 0.72 ± 0.08 | 0.51 ± 0.29 | 0.55 ± 0.16 | 0.60 ± 0.25 | 0.73 ± 0.19 |
| Ours (random-walk grammar GNN) | 0.48 ± 0.27 | 0.70 ± 0.08 | 0.51 ± 0.36 | 0.54 ± 0.13 | 0.49 ± 0.34 | 0.75 ± 0.01 |
| Bag-of-Motifs | 0.40 ± 0.15 | 0.54 ± 0.19 | 0.29 ± 0.08 | 0.52 ± 0.10 | 0.44 ± 0.23 | 0.51 ± 0.25 |
| Bag-of-Motifs (+expert) | 0.68 ± 0.09 | 0.76 ± 0.06 | 0.76 ± 0.07 | 0.66 ± 0.03 | 0.72 ± 0.17 | 0.20 ± 0.51 |
| Bag-of-Motifs (+expert) [raw target] | 0.55 ± 0.25 | 0.74 ± 0.09 | 0.47 ± 0.12 | 0.63 ± 0.06 | 0.70 ± 0.19 | 0.06 ± 0.73 |
| Bag-of-Motifs [raw target] | 0.34 ± 0.09 | 0.54 ± 0.13 | 0.29 ± 0.08 | 0.52 ± 0.10 | 0.44 ± 0.23 | 0.51 ± 0.25 |
| ChemBERTa + ridge | 0.08 ± 0.46 | 0.71 ± 0.08 | -0.07 ± 0.70 | 0.54 ± 0.10 | 0.37 ± 0.30 | 0.56 ± 0.17 |
| ChemBERTa + ridge [raw target] | 0.25 ± 0.20 | 0.74 ± 0.08 | 0.21 ± 0.17 | 0.47 ± 0.13 | 0.37 ± 0.26 | 0.62 ± 0.02 |
| Chemprop D-MPNN | 0.55 ± 0.17 | 0.58 ± 0.12 | 0.56 ± 0.16 | 0.51 ± 0.12 | 0.46 ± 0.02 | 0.69 ± 0.06 |
| Chemprop D-MPNN [raw target] | 0.51 ± 0.13 | 0.60 ± 0.12 | 0.57 ± 0.13 | 0.53 ± 0.09 | 0.47 ± 0.02 | 0.73 ± 0.02 |
| GC regression (motif counts) | 0.48 ± 0.21 | 0.80 ± 0.07 | 0.37 ± 0.17 | 0.27 ± 0.45 | 0.35 ± 0.19 | 0.25 ± 0.41 |
| GC regression (motif counts) [raw target] | 0.37 ± 0.11 | 0.61 ± 0.08 | 0.46 ± 0.27 | 0.31 ± 0.47 | 0.45 ± 0.29 | 0.29 ± 0.43 |
| HM-GNN | -3.23 ± 2.69 | 0.86 ± 0.05 | 0.04 ± 0.19 | 0.61 ± 0.21 | -13.78 ± 18.09 | 0.64 ± 0.18 |
| HM-GNN (+expert) | 0.12 ± 0.81 | -0.37 ± 1.44 | -33.37 ± 47.70 | 0.58 ± 0.25 | -6.18 ± 9.52 | 0.69 ± 0.05 |
| HM-GNN (+expert) [raw target] | -0.53 ± 1.40 | 0.52 ± 0.16 | 0.08 ± 0.29 | 0.52 ± 0.25 | -3.41 ± 5.25 | 0.47 ± 0.21 |
| HM-GNN [raw target] | 0.46 ± 0.14 | 0.42 ± 0.16 | 0.28 ± 0.18 | 0.60 ± 0.09 | 0.23 ± 0.23 | 0.64 ± 0.09 |
| Morgan FP + GP (Tanimoto) | 0.63 ± 0.19 | 0.68 ± 0.11 | 0.63 ± 0.21 | 0.63 ± 0.08 | 0.62 ± 0.22 | 0.57 ± 0.07 |
| Morgan FP + GP (Tanimoto) [raw target] | 0.60 ± 0.16 | 0.64 ± 0.12 | 0.61 ± 0.19 | 0.62 ± 0.07 | 0.62 ± 0.24 | 0.62 ± 0.04 |
| Morgan FP + SVR (RBF) | 0.08 ± 0.65 | 0.72 ± 0.06 | 0.56 ± 0.12 | 0.59 ± 0.08 | 0.39 ± 0.14 | 0.46 ± 0.22 |
| Morgan FP + SVR (RBF) [raw target] | 0.61 ± 0.17 | 0.62 ± 0.07 | 0.55 ± 0.10 | 0.58 ± 0.07 | 0.60 ± 0.26 | 0.45 ± 0.21 |
| Morgan FP + random forest | 0.55 ± 0.21 | 0.66 ± 0.13 | 0.54 ± 0.17 | 0.62 ± 0.09 | 0.61 ± 0.19 | 0.65 ± 0.10 |
| Morgan FP + random forest [raw target] | 0.55 ± 0.21 | 0.66 ± 0.13 | 0.54 ± 0.17 | 0.62 ± 0.09 | 0.61 ± 0.19 | 0.65 ± 0.10 |
| RDKit descriptors + XGBoost | 0.41 ± 0.16 | 0.52 ± 0.35 | 0.63 ± 0.14 | 0.48 ± 0.18 | 0.41 ± 0.39 | 0.87 ± 0.05 |
| RDKit descriptors + XGBoost [raw target] | 0.47 ± 0.27 | 0.28 ± 0.46 | 0.52 ± 0.06 | 0.49 ± 0.18 | 0.18 ± 0.65 | 0.81 ± 0.08 |
| RDKit descriptors + ridge | -436.60 ± 617.89 | -73.82 ± 105.32 | -128.24 ± 181.22 | -1048.10 ± 1482.91 | -1940.84 ± 2743.14 | 0.15 ± 0.72 |
| RDKit descriptors + ridge [raw target] | -436.60 ± 617.89 | -73.94 ± 105.23 | -127.96 ± 181.42 | -1048.13 ± 1482.89 | 0.40 ± 0.22 | -0.10 ± 1.11 |
| Training mean (reference) | -0.06 ± 0.04 | -0.04 ± 0.05 | -0.05 ± 0.03 | -0.09 ± 0.09 | -0.04 ± 0.03 | -0.11 ± 0.08 |
| Training mean (reference) [raw target] | -0.06 ± 0.04 | -0.04 ± 0.05 | -0.05 ± 0.03 | -0.09 ± 0.09 | -0.04 ± 0.03 | -0.11 ± 0.08 |
| polyBERT + GP (RBF) | 0.60 ± 0.05 | 0.66 ± 0.16 | 0.59 ± 0.05 | 0.63 ± 0.14 | 0.68 ± 0.11 | 0.65 ± 0.03 |
| polyBERT + GP (RBF) [raw target] | 0.53 ± 0.07 | 0.59 ± 0.23 | 0.58 ± 0.06 | 0.64 ± 0.14 | 0.68 ± 0.11 | 0.65 ± 0.03 |
| polyBERT + ridge | 0.52 ± 0.39 | 0.66 ± 0.18 | 0.74 ± 0.06 | 0.53 ± 0.09 | 0.74 ± 0.11 | 0.78 ± 0.06 |
| polyBERT + ridge [raw target] | 0.64 ± 0.09 | 0.66 ± 0.11 | 0.66 ± 0.04 | 0.55 ± 0.11 | 0.66 ± 0.13 | 0.68 ± 0.09 |
