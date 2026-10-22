"""
Which trained predictor the paper's numbers come from, in one place.

`conn` is the connection-aware model of campaign/conn_gnn.py, trained and applied by campaign/conn_predict.py under
the same protocol as the published one: the three split seeds, the configuration chosen per gas pair on mean
validation loss across those seeds, the epoch chosen on validation, and an ensemble of the three per-seed models.
It is what Table 2, the screening and the experimental comparison now report.

Setting STEM back to 'published' restores the originally submitted predictor everywhere; both sets of predictions
are kept side by side and nothing is overwritten.
"""
import os

STEM = 'conn'                                    # 'published' for the originally submitted predictor

_CONN = os.path.join('campaign', 'outputs', 'conn')
if STEM == 'conn':
    GRID = os.path.join(_CONN, 'conn_grid.csv')      # per pair x seed x config, for Table 2
    S9 = os.path.join(_CONN, 's9_ensemble.txt')      # the 1000 generated structures
    DATASET = os.path.join(_CONN, 'dataset_ensemble.txt')
    FIG5 = os.path.join(_CONN, 'fig5_ensemble.txt')  # the four synthesized polymers
    CFG = os.path.join(_CONN, 'selected_configs.json')
else:
    GRID = os.path.join('outputs', 'grid_runs.csv')
    S9 = os.path.join('outputs', 's9_ensemble_unique.txt')
    DATASET = os.path.join('outputs', 'dataset_ensemble.txt')
    FIG5 = os.path.join('outputs', 'fig5_ensemble.txt')
    CFG = None
