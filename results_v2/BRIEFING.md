# Briefing: what the v2 results change about the story

For planning the narrative revision together. The numbers in the main text and SI are already updated (see PROVENANCE.md),
and so are factual method descriptions. **Interpretive text was deliberately left as is.** That means the abstract,
teaser, "accurate", "good agreement", "remarkably close" and similar wording. Each item below says what the new
evidence supports, which passages are affected, and the decision it needs from you.

## 1. Prediction accuracy is moderate, not high

| Property | v1 R² (test-selected) | v2 R² (validation-selected, held-out test) | v2 MAE (normalized) |
|---|---|---|---|
| H₂ | 0.69 ± 0.20 | 0.48 ± 0.27 | 0.47 ± 0.21 |
| H₂/N₂ | 0.90 ± 0.04 | 0.70 ± 0.08 | 0.50 ± 0.28 |
| O₂ | 0.82 ± 0.12 | 0.51 ± 0.36 | 0.35 ± 0.19 |
| O₂/N₂ | 0.82 ± 0.09 | 0.54 ± 0.13 | 0.62 ± 0.17 |
| CO₂ | 0.79 ± 0.14 | 0.49 ± 0.34 | 0.37 ± 0.18 |
| CO₂/CH₄ | 0.89 ± 0.03 | 0.75 ± 0.01 | 0.43 ± 0.13 |

- **Why the drop:** v1 chose the epoch, the configuration and the checkpoint by test performance and normalized with all labelled polymers. v2 uses a validation split and training-only statistics. The v1 numbers were optimistic by construction.
- **Large spread across splits:** with about 19 test polymers per split, some splits give R² near 0.1 and others near 0.8. Error bars of ±0.3 on R² are honest but will draw attention.
- **Also corrected:** Table 2's v1 column was labelled MSE but held MAE. It is now labelled MAE, and MSE is in `outputs/table2.txt`.
- **Affected text:** the abstract and teaser ("accurate property prediction"); the Discussion's opening ("an accurate and data-efficient machine learning approach … accurate property prediction"); the Results sentence on data efficiency.
- **Decision:**
  - Reframe as "useful for ranking and screening with small data" rather than "accurate".
  - Optionally add a simple baseline, such as fingerprint ridge regression or random forest under the same splits, so a reader can judge R² ≈ 0.5–0.75. I can run one quickly. If the baseline is close, the story has to rest on the grammar and the interpretability, not on accuracy.

## 2. Ablation: edge weights hurt, directed message passing helps a little

- **Edge weights never did anything in v1.** A code bug discarded them, so the published sentence "it shows that both contributed to better performance" was not supported by any run.
- **With the bug fixed, edge weights worsen the test metrics:** MAE 0.518 / 0.520 against 0.468 / 0.456 without them.
- **Directed message passing helps a little:** MAE 0.456 vs 0.468, R² 0.58 vs 0.51.
- **All three selected models are directed and unweighted.** The new SI Table S10 shows this, and the Methods paragraph has been rewritten to say it.
- **The v1 pooling description was also wrong:** the paper said sum aggregation, but the code uses mean pooling. The Methods text is fixed. The Group Contribution analogy ("in the same spirit as the theory of Group Contribution") was removed with it. If you want to keep the analogy, it needs rewording.
- **Decision:** whether the edge-weight / random-walk-graph message passing stays a selling point. The honest version is: "the walk structure (direction) helps; the transition weights do not."

## 3. The four synthesized polymers: systematic overprediction, ranking partly right

| | PIM-PI-1 | PIM-PI-2 | PIM-PI-3 | PIM-PI-4 |
|---|---|---|---|---|
| H₂ pred / exp | 993 / 470 (2.1×) | 683 / 270 (2.5×) | 1054 / 400 (2.6×) | 970 / 220 (4.4×) |
| O₂ pred / exp | 144 / 110 (1.3×) | 213 / 52 (4.1×) | 169 / 85 (2.0×) | 97 / 40 (2.4×) |
| CO₂ pred / exp | 825 / 660 (1.3×) | 983 / 313 (3.1×) | 878 / 490 (1.8×) | 800 / 227 (3.5×) |
| CO₂/CH₄ pred / exp | 13.3 / 15.8 | 21.8 / 18.6 | 18.3 / 17.3 | 13.7 / 18.3 |
| O₂/N₂ pred / exp | 4.9 / 3.7 | 5.4 / 4.1 | 4.9 / 3.9 | 5.1 / 4.1 |
| H₂/N₂ pred / exp | 23.8 / 16.1 | 28.7 / 21.4 | 28.2 / 18.4 | 24.4 / 22.8 |

- **Every predicted permeability exceeds the measured one**, for all three gases, not just H₂. v1 said "H₂ overpredicted 2–3×". The main text now says 2.1–4.4×, and "within error" has been replaced with actual percentages.
- **Selectivities are much better:**
  - CO₂/CH₄ is within 25% for all four polymers.
  - O₂/N₂ is within 25–32%.
  - H₂/N₂ is within 54%.
- **Rank agreement (Spearman, n = 4, so treat as indicative only):** O₂/N₂ 0.95, CO₂/CH₄ 0.80, H₂ 0.60, H₂/N₂ 0.40, O₂ 0.20, CO₂ 0.20.
- **A likely contributor is the gap between literature and lab values.** The dataset's literature entry for PIM-PI-4's structure (durene PI; the dataset calls it "PIM-PI-1", encoded with L16) lists H₂ 530 / O₂ 150 / CO₂ 1100, but your measurement is 220 / 40 / 227.
  - The model learned from literature values that are 2–5× above what your lab measures for the same polymer.
  - Film history (aging, casting, thermal treatment) is the usual explanation.
  - This is a strong and honest discussion point: the bias is in the training labels, not necessarily in the model.
- **One polymer, two encodings:** the dataset entry (L16 walk) predicts 1255 / 179 / 1394, while our L5 walk for the same polymer predicts 970 / 97 / 800. Different but chemically equivalent walks give predictions that differ by up to 2×, which is worth a sentence or a fix (canonical walks).
- **Affected text:**
  - "in good agreement with the experimentally-determined values"
  - "Overall, the reasonable agreement … highlights the power of the predictive model in accurately predicting"
  - the abstract and teaser claims of experimental validation.
- **Decision:** reframe the validation around selectivity and ranking (which hold up), and present the permeability offset as a calibration issue with an explanation (literature vs lab). Also decide whether to add the durene comparison as a figure or table.

## 4. Generation: valid and novel, but small and less unique than reported

| Metric | v1 | v2 |
|---|---|---|
| Valid (Walk) | 56% | 63% |
| Valid (SMARTS) | 100% | 100% |
| Valid (SMILES) | 91% | 90% |
| Unique | 96% | **62%** |
| Novel | 100% | 100% |
| Diversity | 0.90 | 0.91 |
| Retro* | 38% | 54% |

- **Uniqueness:**
  - v1 measured uniqueness on atom-mapped SMILES, so the same molecule built from different motifs counted as different.
  - On clean canonical SMILES, the 1000 distinct walks give 625 distinct molecules. For example, `P11->P17->P11` and `S20->P17->S20` both build toluene.
  - The new 62% is the honest number. The sentence "produced structures with a high degree of uniqueness" (main text, just above Table 1) needs softening.
- **Size:**
  - 993 of the 1000 best-first structures are two-motif loops (X→Y→X), and the other 7 have three motifs. The median is 16 heavy atoms.
  - The dataset walks average 7.4 motifs.
  - Best-first enumeration favors the shortest, most probable walks, so the "1000 novel structures" are mostly small repeat units, many of them simple substituted aromatics.
  - The Discussion's "number of assembly steps per generated monomer is <4" is now "2.0 (at most 3)", which is true but highlights the issue.
- **Retro\* rose from 38% to 54%.** This is because the molecules are small and have clean SMILES (no atom maps); it is not an improvement in the method.
- **Decision:** this is the biggest story risk. Options:
  - (a) Report it as is, and describe best-first as producing building-block-scale candidates.
  - (b) Add a length-controlled or random-sampling generation set for Table 1, which already supplies the validity numbers.
  - (c) Change best-first to require at least k motifs.
  - Any of these is a few hours of compute. (b) is the cheapest and arguably what Table 1 should use anyway.

## 5. Screening (SI Figs S2–S4)

- **Structures within the error thresholds:**
  - H₂/N₂: 372 (v1 29)
  - O₂/N₂: 3 (v1 33)
  - CO₂/CH₄: 378 (v1 82)
- **The O₂ collapse:** the O₂ std threshold (100 barrer) is below the ensemble's median O₂ std (≈250), because the v2 models disagree more across splits. Either justify the thresholds or scale them, for example a fixed percentile of the ensemble std.
- **Fewer structures beyond the upper bound:** most high-confidence novel structures sit near or below the 2008 upper bound. Revisit "many of them show significant promise".
- **Two dataset polymers get negative predicted H₂ permeability** (they are left off the log-scale plots). It's a minor point, but it argues for predicting log-permeability. That is a standard choice, would also tame the overprediction, and is a modelling change that could be made before submission.
- **The motif-usage paragraph is regenerated.**
  - The most common motifs are P6, P27 and S10 (H₂/N₂ and CO₂/CH₄).
  - The "S20→S21 example" claim in v1 was not supported and is gone.
- **Table S1 (preliminary screening) is unchanged, as agreed.** Its caption should say it comes from an earlier model version.

## 6. Consistency issues still open (need your call)

1. **Fig 4C swaps PIM-PI-2 and PIM-PI-3** relative to Fig 5, the SI tables and the NMR. I kept the Fig 5 / NMR mapping. Fig 4C needs redrawing, or its labels swapping.
2. **Name collisions:** the dataset already contains polymers named PIM-PI-1 … PIM-PI-11 (Table S6/S7), and our four synthesized ones reuse PIM-PI-1 to 4. The dataset's "PIM-PI-1" is our PIM-PI-4. Suggest renaming ours (for example PIM-PI-G1 to G4) or adding a footnote.
3. **Fig S1 image** still shows the pre-overhaul groups for 13 entries: 11 edited, plus P42 and S33 new, plus P36's moved R site. It needs redrawing. The caption already says 99 groups.
4. **Main-text walk notation** (for example "S32:1") is out of date compared with the edge-id notation in Table S6.
5. **The R-site rule rejects about 9% of syntactically valid walks** (no legal attachment). Masking illegal transitions during sampling would raise Valid (Walk). Should this be mentioned or implemented?
6. **SI Table S7 renders with wrapped numbers in LibreOffice.** This happens in the original too, and may be fine in Word. I did not change it.

## 7. Suggested order for our planning conversation

1. Decide the generation framing (§4), since it may mean new runs.
2. Decide whether to add a baseline and/or log-permeability (§1, §5), since that also means new runs.
3. Rewrite the abstract, teaser and discussion claims around: grammar → valid, novel, synthesizable building blocks; predictor → selectivity and ranking; literature-vs-lab calibration as a finding.
4. Fix the figure and label issues (§6).
