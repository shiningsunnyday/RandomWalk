"""
Mechanism behind the cross-split disagreement, measured on the FROZEN grid (results_v2/outputs/pred/), at zero
compute cost. For each (gas pair, split seed) it reconstructs the test split with analysis/splits.py
(the frozen runs' test_idx.txt is corrupted, so the split is recomputed from the seed; positions are mapped
back through the walks_file line order) and relates the
test split's target geometry to the test R^2 averaged over all 48 configurations of that pair and seed.

usage: python analysis/why_variance.py [> analysis/why_variance.txt]
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import test_positions
A = os.path.dirname(os.path.abspath(__file__)); RV = os.path.join(os.path.dirname(A), '..')
GC = os.path.join(RV, 'data', 'group-contrib')
lines = [l for l in open(os.path.join(GC, 'polymer_walks_v2_preprocess.txt')).read().split('\n') if l.strip()]
f1 = lambda x: float('nan') if x.strip() in ('-', '_', '') else float(x.strip())
props = np.array([[f1(x) for x in l.split()[-1].strip('()').split(',')] for l in lines])
PAIR = {'H2_N2': (0, 1), 'O2_N2': (2, 1), 'CO2_CH4': (4, 3)}
R = pd.read_csv(os.path.join(RV, 'outputs', 'grid_runs.csv'))
rows = []
srcs = {}
for _, r in R.iterrows():
    pair = r['pair']; a, b = PAIR[pair]
    sl = [i for i in range(len(lines)) if np.isfinite(props[i, a]) and np.isfinite(props[i, b])]
    d = r['dir'] if os.path.isabs(r['dir']) else os.path.join(RV, r['dir'])
    ti, _src = test_positions(d, len(sl), int(r['seed']), 'none')
    srcs[_src] = srcs.get(_src, 0) + 1
    li = [sl[p] for p in ti]; tr = [sl[p] for p in range(len(sl)) if p not in set(ti)]
    perm = props[li, a]; gas = pair.split('_')[0]
    rows.append(dict(pair=pair, seed=r['seed'],
                     r2_perm=r[f'{gas}|r^2'], r2_sel=r[f'{pair.replace("_", "/")}|r^2'],
                     test_log_sd=np.std(np.log10(perm)), test_raw_sd=np.std(perm),
                     test_logrange=np.log10(perm.max() / perm.min()),
                     max_over_trainmax=perm.max() / props[tr, a].max()))
D = pd.DataFrame(rows)
g = D.groupby(['pair', 'seed']).agg(r2_perm=('r2_perm', 'mean'), r2_sel=('r2_sel', 'mean'),
                                    test_log_sd=('test_log_sd', 'first'), test_logrange=('test_logrange', 'first'),
                                    test_raw_sd=('test_raw_sd', 'first'),
                                    max_over_trainmax=('max_over_trainmax', 'first'))
print('test-split source (file = trusted test_idx.txt, recomputed = from the seed):', srcs, '\n')
print('test-split target geometry vs frozen test R^2, averaged over the 48 configurations of each (pair, seed)\n')
print(g.round(3).to_string(), '\n')
for c in ['test_log_sd', 'test_logrange', 'test_raw_sd', 'max_over_trainmax']:
    print(f'corr(mean test R2, {c:20s}) permeability {np.corrcoef(g.r2_perm, g[c])[0,1]:+.2f}'
          f'   selectivity {np.corrcoef(g.r2_sel, g[c])[0,1]:+.2f}')
print('''
Reading (corrected 2026-10-19 22:10, after the frozen test_idx.txt bug was found and the splits were
recomputed from the seed):
  * max_over_trainmax is 0.43-0.67 for ALL nine frozen (pair, seed) cells, so NO frozen test split contains a
    polymer more permeable than the training maximum. The earlier claim that the collapsing splits were the
    extrapolating ones came from the corrupted test_idx.txt and is withdrawn.
  * Every correlation is now positive: the splits with the SMALLEST test-split target spread are the ones whose
    mean test R^2 collapses. H2/N2 seed 2 has the smallest log spread (0.717) and the smallest raw spread
    (2019) of its pair and gives mean R^2 = -1.03; O2/N2 seed 2 has the smallest raw spread (819) of its pair
    and gives -0.62.
  * Mechanism: R^2 = 1 - SSE/SST, and SST is the test split's own target variance. With ~19 test polymers a
    split that happens to be narrow has a tiny denominator, so an ordinary absolute error turns into a large
    negative R^2. The disagreement is as much a property of the metric on small, unequal test splits as of the
    model.
  * This is a direct argument for stratifying the split on the target quantile: it equalises SST across splits.
    It is also an argument for the log target, which compresses a 3-4 decade range into one where the
    split-to-split variation in SST is much smaller.''')
