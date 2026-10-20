"""
Does our standing among the baselines survive the honest protocol? -> outputs/k10.md, outputs/k10.json

The campaign showed that three split seeds is too few to estimate R2: the same protocol at ten seeds gives our model
mean test R2 0.34, not 0.58 (BRIEFING_2 §6). That raises the question decision 0 turns on -- whether the three-seed
numbers flatter *us* or flatter *everyone*. This script answers it by putting the classical baselines on the same ten
seeds the campaign used (`bl_classical.py --seeds 1,...,10 --out outputs/results_k10.jsonl`) and tabulating the two
protocols side by side. Our own K=10 row comes from the campaign's `base` arm, selected the same way as everywhere
else: per (property, seed), the run with the lowest validation loss.

Nothing here touches outputs/results.jsonl or the SI tables.
usage: python baselines/bl_k10.py
"""
import os, json
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(H, 'outputs')
PROPS = ['H2', 'H2/N2', 'O2', 'O2/N2', 'CO2', 'CO2/CH4']
OURS = 'Ours (random-walk grammar GNN)'


def load(path):
    """baseline -> property -> (mae, r2) means over the seeds in that file"""
    rows, out = {}, {}
    for line in open(path):
        r = json.loads(line)
        rows[(r['baseline'], r['pair'])] = r                 # a re-run supersedes an earlier row
    for (name, _), r in rows.items():
        for j, prop in enumerate(r['targets']):
            out.setdefault(name, {})[prop] = (r['mae_mean'][j], r['r2_mean'][j], len(r['per_seed']))
    return out


def ours_k3():
    """our frozen three-split numbers, from the predictor grid (bl_summarize.ours), not from results.jsonl"""
    R = pd.read_csv(os.path.join(os.path.dirname(H), 'outputs', 'grid_runs.csv'))
    CFG = ['ew', 'fw', 'init', 'directed']
    out = {}
    for pair, props in (('H2_N2', ('H2', 'H2/N2')), ('O2_N2', ('O2', 'O2/N2')), ('CO2_CH4', ('CO2', 'CO2/CH4'))):
        g = R[R.pair == pair]
        m = g.groupby(CFG)['val_loss'].agg(['mean', 'count'])
        s = g[(g[CFG] == pd.Series(dict(zip(CFG, m[m['count'] == 3]['mean'].idxmin())))).all(axis=1)]
        for p in props:
            out[p] = (float(s[f'{p}|mae'].mean()), float(s[f'{p}|r^2'].mean()), len(s))
    return out


def ours_k10(arm='base'):
    """a campaign arm at ten splits: same architecture and protocol, the arm's one change"""
    c = os.path.join(os.path.dirname(H), 'campaign', 'outputs', 'campaign_runs.csv')
    if not os.path.exists(c):
        return None
    d = pd.read_csv(c)
    d = d[d.arm == arm]
    if d.empty:
        return None
    out = {}
    for p in PROPS:
        per = []
        for seed, g in d[d[f'{p}|r^2'].notna()].groupby('seed'):
            r = g.loc[g['val_loss'].idxmin()]
            per.append((r[f'{p}|mae'], r[f'{p}|r^2']))
        out[p] = (float(np.mean([x[0] for x in per])), float(np.mean([x[1] for x in per])), len(per))
    return out


k3, k10 = load(os.path.join(O, 'results.jsonl')), load(os.path.join(O, 'results_k10.jsonl'))
k3[OURS] = ours_k3()
o10 = ours_k10()
if o10:
    k10[OURS] = o10
# the baselines are tuned, so `log_target` is in their grid; our published row is raw-target only. The campaign's
# `log` and `combo_y2` arms are the like-for-like comparison, and they are what decision 0 would adopt.
EXTRA = {'Ours, log target (campaign `log`)': 'log', 'Ours, log + stratified (campaign `combo_y2`)': 'combo_y2'}
for lab, arm in EXTRA.items():
    a = ours_k10(arm)
    if a:
        k10[lab] = a
        k3[lab] = k3[OURS]            # no K=3 run of these arms exists; the frozen row is the thing they replace


def agg(d):
    mae = [d[p][0] for p in PROPS if p in d]
    r2 = [d[p][1] for p in PROPS if p in d]
    if len(mae) < len(PROPS):
        return None
    return {'mae': float(np.mean(mae)), 'r2': float(np.mean(r2)), 'median_r2': float(np.median(r2)),
            'worst_r2': float(np.min(r2)), 'seeds': d[PROPS[0]][2]}


A = {n: {'k3': agg(k3.get(n, {})), 'k10': agg(k10.get(n, {}))} for n in set(k3) | set(k10)}
A = {n: v for n, v in A.items() if v['k3'] and v['k10']}
json.dump(A, open(os.path.join(O, 'k10.json'), 'w'), indent=1)

order = sorted(A, key=lambda n: A[n]['k10']['mae'])
rank3 = sorted([n for n in A if n not in EXTRA], key=lambda n: A[n]['k3']['mae'])   # the arms have no K=3 run
L = ['# The baselines at ten split seeds', '',
     f'{len(A)} methods have both protocols. Seeds 1-3 (frozen, Tables S12/S13) against seeds 1-10, everything else',
     'identical. `bl_k10.py`; our own K=10 row is the campaign `base` arm.', '',
     '| Method | MAE K=3 | MAE K=10 | R² K=3 | R² K=10 | ΔR² | rank by MAE (K=3 → K=10) |', '|---|---|---|---|---|---|---|']
for n in order:
    a, b = A[n]['k3'], A[n]['k10']
    nm = f'**{n}**' if n.startswith('Ours') else n
    k3cell = '—' if n in EXTRA else f"{a['mae']:.3f}"
    r3cell = '—' if n in EXTRA else f"{a['r2']:.3f}"
    dcell = '—' if n in EXTRA else f"{b['r2'] - a['r2']:+.3f}"
    L.append(f"| {nm} | {k3cell} | {b['mae']:.3f} | {r3cell} | {b['r2']:.3f} | {dcell} | "
             f"{'—' if n in EXTRA else rank3.index(n) + 1} → {order.index(n) + 1} |")
# the diverged ridge fit (R2 in the hundreds) would swamp a mean, and the campaign arms have no K=3 run
real = [n for n in A if n not in EXTRA and A[n]['k3']['r2'] > -1]
dr = [A[n]['k10']['r2'] - A[n]['k3']['r2'] for n in real]
tau = pd.Series([rank3.index(n) for n in real]).corr(pd.Series([order.index(n) for n in real]), method='kendall')
L += ['', f'Over the {len(real)} methods with both protocols and no diverged fit: median ΔR² {np.median(dr):+.3f}, '
          f'mean {np.mean(dr):+.3f}, and {sum(x < 0 for x in dr)} of {len(dr)} lose R² at ten seeds. MAE moves by at '
          f'most {max(abs(A[n]["k10"]["mae"] - A[n]["k3"]["mae"]) for n in real):.3f}.',
      f"Kendall tau between the two MAE rankings: {tau:.2f}. Our rank by MAE goes "
      f"{rank3.index(OURS) + 1} → {order.index(OURS) + 1} of {len(order)} rows.",
      f"Our own R² drop is {A[OURS]['k10']['r2'] - A[OURS]['k3']['r2']:+.3f} against a median of {np.median(dr):+.3f}, "
      f"the {sorted(dr).index(A[OURS]['k10']['r2'] - A[OURS]['k3']['r2']) + 1}{'st' if sorted(dr).index(A[OURS]['k10']['r2'] - A[OURS]['k3']['r2']) == 0 else 'nd' if sorted(dr).index(A[OURS]['k10']['r2'] - A[OURS]['k3']['r2']) == 1 else 'th'}"
      f" largest of the {len(dr)}: at the raw target our across-split spread is among the widest in the table, which "
      f"is the campaign's finding restated. The `log` arm removes it."]
open(os.path.join(O, 'k10.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
