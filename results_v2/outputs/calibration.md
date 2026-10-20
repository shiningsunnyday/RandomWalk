# Calibrating on the one validated polymer that is in the dataset

Reference: our **PIM-PI-4** = dataset row **PIM-PI-1**. Calibration factor is literature / measured on that
polymer alone; the three novel polymers contributed nothing to it.

| property | literature | our measurement | calibration factor |
|---|---|---|---|
| H2 | 530 | 220 | **2.41×** |
| O2 | 150 | 40 | **3.75×** |
| CO2 | 1.1e+03 | 227 | **4.85×** |
| H2/N2 | 11.3 | 22.8 | **0.49×** |
| O2/N2 | 3.19 | 4.1 | **0.78×** |
| CO2/CH4 | 14.3 | 18.3 | **0.78×** |

## The three novel polymers: predicted / measured, before and after

| polymer | H2 | O2 | CO2 | H2/N2 | O2/N2 | CO2/CH4 |
|---|---|---|---|---|---|---|
| **PIM-PI-1** raw | 2.11 | 1.31 | 1.25 | 1.48 | 1.32 | 0.84 |
| **PIM-PI-1** calibrated | 0.88 | 0.35 | 0.26 | 2.99 | 1.69 | 1.08 |
| **PIM-PI-2** raw | 2.53 | 4.13 | 3.14 | 1.34 | 1.31 | 1.17 |
| **PIM-PI-2** calibrated | 1.05 | 1.10 | 0.65 | 2.71 | 1.69 | 1.50 |
| **PIM-PI-3** raw | 2.63 | 1.99 | 1.79 | 1.53 | 1.25 | 1.06 |
| **PIM-PI-3** calibrated | 1.09 | 0.53 | 0.37 | 3.10 | 1.61 | 1.35 |
| PIM-PI-4 *(reference)* raw | 4.41 | 2.42 | 3.52 | 1.07 | 1.25 | 0.75 |
| PIM-PI-4 *(reference)* calibrated | 1.83 | 0.64 | 0.73 | 2.16 | 1.61 | 0.96 |

## Per property, over the three novel polymers

| property | factor c | geomean ratio raw | geomean ratio calibrated | calibrated range | within error |
|---|---|---|---|---|---|
| H2 | 2.41× | 2.41 | **1.00** | 0.88–1.09 | 0/3 → **3/3** |
| O2 | 3.75× | 2.21 | **0.59** | 0.35–1.10 | 1/3 → **1/3** |
| CO2 | 4.85× | 1.92 | **0.40** | 0.26–0.65 | 1/3 → **0/3** |
| H2/N2 | 0.49× | 1.45 | **2.93** | 2.71–3.10 | 1/3 → **0/3** |
| O2/N2 | 0.78× | 1.29 | **1.66** | 1.61–1.69 | 1/3 → **0/3** |
| CO2/CH4 | 0.78× | 1.02 | **1.30** | 1.08–1.50 | 2/3 → **1/3** |

## Is the systematic bias removed? (three novel polymers only)

| | geometric-mean ratio | mean &#124;log₁₀ ratio&#124; | within error | range |
|---|---|---|---|---|
| permeability, raw | 2.17 | 0.336 | 2/9 | 1.25–4.13 |
| permeability, **calibrated** | **0.62** | **0.233** | **4/9** | 0.26–1.10 |
| selectivity, raw | 1.24 | 0.110 | 4/9 | 0.84–1.53 |
| selectivity, **calibrated** | **1.85** | **0.267** | **1/9** | 1.08–3.10 |
