"""
Table 2, the SI edge-weight x directedness ablation, and the ensemble members, from the v2 predictor grid
(results_v2/outputs/pred/). Selection never looks at the test split:

  checkpoint   each run is used at its lowest validation loss (the epoch its predictor_ckpt was saved at)
  model        per gas pair, ONE configuration (edge weights, W features, init, direction): the one with the lowest
               validation loss averaged over the three splits; that configuration's run on each split is the model for
               that split, and its test (heldout_*) metrics at the checkpoint epoch are reported
  Table 2      mean +- std (population) over the 3 split seeds of test MAE, MSE, R2, normalized scale
  ablation     same rule restricted to each (directed, edge_weights) combination
  ensembles    the selected model per gas pair and split seed -> outputs/ensemble_<pair>.txt

usage: python results_v2/summarize.py
"""
import os, json, glob
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
PAIRS = {(0, 1): ('H2', 'H2_N2'), (2, 1): ('O2', 'O2_N2'), (4, 3): ('CO2', 'CO2_CH4')}

runs = []
for cf in sorted(glob.glob(os.path.join(H, 'outputs', 'pred', '*', 'predictor_*', 'config.json'))):
    d = os.path.dirname(cf); c = json.load(open(cf)); c = json.loads(c) if isinstance(c, str) else c
    m = os.path.join(d, 'metrics.csv')
    if not os.path.exists(m): continue
    df = pd.read_csv(m, index_col=0)
    if len(df) < c['num_epochs']: continue
    e = int(df['val_loss'].values.argmin())
    gas, pair = PAIRS[tuple(c['property_cols'])]
    rec = dict(dir=d, pair=pair, seed=c['test_seed'], ew=bool(c['edge_weights']), fw=bool(c['feat_concat_W']),
               init=c['init'], directed=bool(c.get('ablate_bidir')), epoch=e, val_loss=float(df['val_loss'].iloc[e]))
    for prop, col in ((gas, f'permeability_{gas}'), (pair.replace('_', '/'), f'selectivity_{pair}')):
        for k in ('mae', 'mse', 'r^2'):
            rec[f'{prop}|{k}'] = float(df[f'heldout_{col}_{k}'].iloc[e])
    runs.append(rec)
R = pd.DataFrame(runs)
print(f'{len(R)} complete runs')


CFG = ['ew', 'fw', 'init', 'directed']


def select(sub):
    """per gas pair, the configuration with the lowest validation loss averaged over the 3 splits; its run per split"""
    out = []
    for pair, g in sub.groupby('pair'):
        m = g.groupby(CFG)['val_loss'].agg(['mean', 'count'])
        best = m[m['count'] == 3]['mean'].idxmin()
        out.append(g[(g[CFG] == pd.Series(dict(zip(CFG, best)))).all(axis=1)])
    return pd.concat(out)


def table(sel, title):
    lines = [title, f'{"property":10s} {"MAE":>13s} {"MSE":>13s} {"R2":>13s}']
    for pair in ('H2_N2', 'O2_N2', 'CO2_CH4'):
        s = sel[sel.pair == pair]
        for prop in (pair.split('_')[0], pair.replace('_', '/')):
            v = {k: s[f'{prop}|{k}'].values for k in ('mae', 'mse', 'r^2')}
            lines.append(f'{prop:10s} ' + ' '.join(f'{v[k].mean():6.2f} ± {v[k].std():4.2f}' for k in ('mae', 'mse', 'r^2'))
                         + f'   (n={len(s)})')
    return lines


sel = select(R)
out = table(sel, 'Table 2: test metrics of the validation-selected model per gas pair and split (normalized scale)')
out.append('selected models:')
for _, r in sel.sort_values(['pair', 'seed']).iterrows():
    out.append(f"  {r.pair:8s} seed {r.seed}: epoch {r.epoch:4d} val_loss {r.val_loss:.3f} ew={r.ew} fw={r.fw} init={r.init} "
               f"directed={r.directed}  {os.path.relpath(r.dir, H)}")
out.append('')
out.append('SI ablation (edge weights x message-passing direction; other settings selected on validation):')
out.append(f'{"setting":30s} {"mean test MAE":>14s} {"mean test R2":>13s}')
for directed in (False, True):
    for ew in (False, True):
        s = select(R[(R.directed == directed) & (R.ew == ew)])
        mae = s[[c for c in s.columns if c.endswith('|mae')]].mean(axis=1).values   # per (pair, seed): mean over its 2 props
        r2 = s[[c for c in s.columns if c.endswith('|r^2')]].mean(axis=1).values
        name = f"{'directed' if directed else 'undirected'}, {'with' if ew else 'without'} edge weights"
        out.append(f'{name:30s} {np.nanmean(mae):6.3f} ± {np.nanstd(mae):5.3f} {np.nanmean(r2):6.3f} ± {np.nanstd(r2):5.3f}')
        out += ['    ' + l for l in table(s, '')[2:]]
open(os.path.join(H, 'outputs', 'table2.txt'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
for pair in ('H2_N2', 'O2_N2', 'CO2_CH4'):
    dirs = sel[sel.pair == pair].sort_values('seed')['dir'].tolist()
    open(os.path.join(H, 'outputs', f'ensemble_{pair}.txt'), 'w').write('\n'.join(dirs) + '\n')
R.to_csv(os.path.join(H, 'outputs', 'grid_runs.csv'), index=False)

# ---- machine-readable ablation for SI Table S10
rows = []
# The default configuration (directed message passing, no edge weights) is the first row; the other three rows are
# what the ablation removes from it, so the table reads top-down as "default, then each thing changed".
for directed, ew in ((True, False), (False, False), (True, True), (False, True)):
    s = select(R[(R.directed == directed) & (R.ew == ew)])
    mae = s[[c for c in s.columns if c.endswith('|mae')]].mean(axis=1).values
    r2 = s[[c for c in s.columns if c.endswith('|r^2')]].mean(axis=1).values
    name = 'directed (walk order)' if directed else 'undirected'
    rows.append([name + (' — default' if (directed and not ew) else ''), 'yes' if ew else 'no',
                 f'{np.mean(mae):.3f} ± {np.std(mae):.3f}', f'{np.mean(r2):.3f} ± {np.std(r2):.3f}'])
json.dump({'header': ['Message passing', 'Edge weights', 'Test MAE', 'Test R²'], 'rows': rows,
           'caption': 'Table S10: Ablation of the predictor\'s message-passing direction and edge weights. The first '
                      'row is the default configuration used throughout the paper, in which messages flow along the '
                      'direction of the random walk and the aggregation is unweighted; the remaining rows change one '
                      'or both of those choices. For each setting, the remaining options (motif-graph features, '
                      'initialization) are selected on the validation set per gas pair (mean validation loss over the '
                      'three splits); we report the test MAE and R² on normalized values, averaged over the six '
                      'properties, as mean ± standard deviation over the three gas pairs and three train-test '
                      'splits. The difference between the first two rows is within that spread, and repeating the '
                      'comparison on ten train/test splits does not resolve a difference in either direction, so '
                      'the direction of message passing is chosen for the reason given in the Methods rather than '
                      'on these numbers; the edge-weight rows are a much larger effect.'},
          open(os.path.join(H, 'outputs', 'ablation.json'), 'w'), indent=1)
