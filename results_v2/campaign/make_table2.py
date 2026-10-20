"""
Table 2 for one campaign arm, in the format of results_v2/outputs/table2.txt, plus the raw-barrer-space
counterpart so a reader can compare a log-target arm with the frozen raw-target numbers on equal terms.

usage: python make_table2.py <arm>   -> campaign/outputs/table2_<arm>.txt
"""
import os, sys, json
import numpy as np, pandas as pd
import summarize_campaign as S
H = os.path.dirname(os.path.abspath(__file__))
arm = sys.argv[1]
R = pd.read_csv(os.path.join(H, 'outputs', 'campaign_runs.csv'))
g = R[R.arm == arm]
assert len(g), arm
sel = S.select(g)
L = [f'Table 2 (campaign arm `{arm}`): test metrics of the validation-selected model per gas pair and split.',
     f'K = {sel.groupby("pair").size().max()} split seeds. Selection: lowest validation loss for the epoch and the',
     'checkpoint; lowest mean validation loss over the splits for the configuration. The test split is only evaluated.',
     '']
logt = bool(sel.logt.iloc[0])
for space, suffix in ((('log10' if logt else 'raw') + '-target space (the space the model was trained in)', ''),
                      ('raw barrer space (log-target predictions inverted with 10^x)', 'raw')):
    L.append(f'-- {space}')
    L.append(f'{"property":10s} {"MAE":>13s} {"MSE":>13s} {"R2":>13s} {"Spearman":>13s}')
    for pair in ('H2_N2', 'O2_N2', 'CO2_CH4'):
        s = sel[sel.pair == pair]
        for prop in (pair.split('_')[0], pair.replace('_', '/')):
            if suffix == '':
                keys = [('mae', 'mae'), ('mse', 'mse'), ('r^2', 'r^2'), ('rho', 'rho')]
            else:
                keys = [('mae', 'rawmae'), ('mse', None), ('r^2', 'rawr2'), ('rho', 'rho')]
            cells = []
            for _, k in keys:
                if k is None:
                    cells.append(f'{"-":>13s}'); continue
                v = s[f'{prop}|{k}'].values
                cells.append(f'{np.nanmean(v):6.2f} ± {np.nanstd(v):4.2f}')
            L.append(f'{prop:10s} ' + ' '.join(cells) + f'   (n={len(s)})')
    L.append('')
L.append('selected models (one per split seed; these are the ensemble members):')
for _, r in sel.sort_values(['pair', 'seed']).iterrows():
    L.append(f'  {r.pair:8s} seed {r.seed:2d}: epoch {r.epoch:4d} val_loss {r.val_loss:.3f} fw={r.fw} '
             f'logt={r.logt} stratify={r.stratify}/{r.strata} vsmooth={r.vsmooth}  {os.path.relpath(r["dir"], H)}')
txt = '\n'.join(L)
open(os.path.join(H, 'outputs', f'table2_{arm}.txt'), 'w').write(txt + '\n')
print(txt)
