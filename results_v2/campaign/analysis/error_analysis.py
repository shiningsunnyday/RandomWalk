"""
Error analysis by chemical subspace, for one campaign arm. Answers Reviewer 1's "where does the model fail?
Are there systematic errors in certain chemical subspaces?"

Uses **out-of-fold** predictions only: for each labelled polymer and each split seed in which that polymer
landed in the test split, the prediction of that split's member. Nothing in this file is an in-sample residual.
Reported per subspace (fingerprint cluster, grouped Murcko scaffold, motif-count tertile, name family,
property tertile):

  n                     out-of-fold polymer x property residuals in the subspace
  median signed log10 ratio     log10(pred / measured): > 0 = systematic over-prediction
  median |log10 ratio|          typical multiplicative error
  R^2 and Spearman rho of the out-of-fold predictions within the subspace

usage: python analysis/error_analysis.py <arm> [<arm> ...]
  needs campaign/outputs/screen_<arm>_dataset_members.csv (screening.py) and selected_<arm>.csv
  The test split of each member comes from analysis/splits.py:test_positions, which refuses to trust the
  frozen runs' corrupted test_idx.txt and recomputes the split from the seed instead (see splits.py).
  writes campaign/analysis/error_<arm>.json and campaign/analysis/figs/error_<arm>.png
"""
import os, sys, json, csv, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splits import test_positions

A = os.path.dirname(os.path.abspath(__file__))
H = os.path.dirname(A)
GC = os.path.join(H, '..', 'data', 'group-contrib')
F = os.path.join(A, 'figs'); os.makedirs(F, exist_ok=True)
GASES = ['H2', 'N2', 'O2', 'CH4', 'CO2']
PROPS = ['H2', 'H2/N2', 'O2', 'O2/N2', 'CO2', 'CO2/CH4']
PAIRCOLS = {'H2_N2': (0, 1), 'O2_N2': (2, 1), 'CO2_CH4': (4, 3)}

lines = [l.rstrip('\n') for l in open(os.path.join(GC, 'polymer_walks_v2_preprocess.txt')) if l.strip()]
names = [l.split()[0] for l in lines]
walks = [re.sub(r'-\d+>', '->', l.split()[1]) for l in lines]
f1 = lambda x: float('nan') if x.strip() in ('-', '_', '') else float(x.strip())
props = np.array([[f1(x) for x in l.split()[-1].strip('()').split(',')] for l in lines])
n = len(lines)

truth = {}
for i, w in enumerate(walks):
    v = props[i]
    truth[(i, 'H2')] = v[0]; truth[(i, 'O2')] = v[2]; truth[(i, 'CO2')] = v[4]
    truth[(i, 'H2/N2')] = v[0] / v[1]; truth[(i, 'O2/N2')] = v[2] / v[1]; truth[(i, 'CO2/CH4')] = v[4] / v[3]

# subspace labels per walks_file line
subs = {}
for fn, lab in (('fingerprint cluster', 'strata_fpclust.txt'), ('Murcko scaffold group', 'strata_scaffold.txt'),
                ('motif-count tertile', 'strata_struct.txt'), ('H2 permeability tertile', 'strata_y_01.txt')):
    subs[fn] = [int(x) for x in open(os.path.join(A, lab)).read().split()]


def family(nm):
    s = nm.upper()
    if 'KAUST' in s: return 'KAUST-PI'
    if 'PIM' in s and 'PI-' in s: return 'PIM-PI'
    if 'PIM' in s: return 'PIM'
    if re.match(r'^(TPDA|TBDA|SPDA|PMDA|BPDA|TDA|ODA|BTDA|BPADA|SBIDA|SBFDA|PPIMDA|EAD)', s):
        return 'polyimide (dianhydride series)'
    if 'PI' in s: return 'polyimide (other)'
    if s[0].isdigit() or s[0] == '-': return 'numbered entry'
    return 'other'


fams = [family(x) for x in names]
subs['name family'] = fams


def subset_lines(pair):
    """walks_file line indices that have both properties of this pair, in line order (= dags_copy order)"""
    a, b = PAIRCOLS[pair]
    return [i for i in range(n) if np.isfinite(props[i, a]) and np.isfinite(props[i, b])]


def run(arm):
    sel = list(csv.DictReader(open(os.path.join(H, 'outputs', f'selected_{arm}.csv'))))
    mem = list(csv.DictReader(open(os.path.join(H, 'outputs', f'screen_{arm}_dataset_members.csv'))))
    pred = {(r['dir'], r['walk'], r['prop']): float(r['pred']) for r in mem}
    # out-of-fold residuals
    oof = []
    srcs = {}
    for r in sel:
        pair, d = r['pair'], r['dir']
        sl = subset_lines(pair)
        cfg = json.load(open(os.path.join(d, 'config.json')))
        cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
        ti, src = test_positions(d, len(sl), int(cfg['test_seed']), cfg.get('stratify', 'none'))
        srcs[src] = srcs.get(src, 0) + 1
        for p in ti:
            li = sl[p]
            for prop in (pair.split('_')[0], pair.replace('_', '/')):
                t = truth[(li, prop)]
                k = (d, walks[li], prop)
                if k not in pred or not np.isfinite(t) or t <= 0: continue
                oof.append(dict(line=li, name=names[li], prop=prop, seed=int(r['seed']), true=t, pred=pred[k]))
    out = {'arm': arm, 'test_split_source': srcs, 'n_oof': len(oof),
           'n_polymers_covered': len({(o['line'], o['prop']) for o in oof}),
           'mean_times_in_test': len(oof) / max(1, len({(o['line'], o['prop']) for o in oof}))}

    def stats(rows):
        if len(rows) < 3: return None
        t = np.array([r['true'] for r in rows]); p = np.array([r['pred'] for r in rows])
        ok = p > 0
        lr = np.full(len(rows), np.nan); lr[ok] = np.log10(p[ok] / t[ok])
        return dict(n=len(rows), n_nonpositive_pred=int((~ok).sum()),
                    median_signed_log_ratio=float(np.nanmedian(lr)),
                    median_abs_log_ratio=float(np.nanmedian(np.abs(lr))),
                    frac_over=float(np.nanmean(lr > 0)),
                    r2_log=float(r2_score(np.log10(t[ok]), np.log10(p[ok]))) if ok.sum() > 2 else None,
                    rho=float(spearmanr(t, p).correlation))

    out['overall'] = {p: stats([o for o in oof if o['prop'] == p]) for p in PROPS}
    out['overall_all'] = stats(oof)
    out['by_subspace'] = {}
    for sname, lab in subs.items():
        d = {}
        for k in sorted(set(lab)):
            rows = [o for o in oof if lab[o['line']] == k]
            s = stats(rows)
            if s: d[str(k)] = s
        out['by_subspace'][sname] = d
    # the worst 12 individual polymers by median |log ratio| over their properties
    byp = {}
    for o in oof:
        if o['pred'] > 0: byp.setdefault((o['name'], o['line']), []).append(abs(np.log10(o['pred'] / o['true'])))
    worst = sorted(((float(np.median(v)), k[0], k[1]) for k, v in byp.items()), reverse=True)[:12]
    out['worst_polymers'] = [{'name': a, 'line': b, 'median_abs_log_ratio': round(m, 3)} for m, a, b in worst]
    json.dump(out, open(os.path.join(A, f'error_{arm}.json'), 'w'), indent=1)

    # figure: signed log-ratio distribution per subspace
    fig, axes = plt.subplots(1, len(subs), figsize=(4.0 * len(subs), 4.2), squeeze=False)
    for ax, (sname, lab) in zip(axes[0], subs.items()):
        keys = sorted(set(lab), key=str)
        data, labs = [], []
        for k in keys:
            v = [np.log10(o['pred'] / o['true']) for o in oof if lab[o['line']] == k and o['pred'] > 0]
            if len(v) >= 3: data.append(v); labs.append(f'{k}\n(n={len(v)})')
        if data:
            ax.boxplot(data, labels=labs, showfliers=False)
            ax.axhline(0, c='r', lw=1)
        ax.set_title(sname, fontsize=10); ax.set_ylabel('log10(pred / measured)')
        ax.tick_params(axis='x', labelsize=7, rotation=45)
    fig.suptitle(f'Out-of-fold error by chemical subspace — arm `{arm}`', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(F, f'error_{arm}.png'), dpi=200); plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != 'by_subspace'}, indent=1))
    for sname, d in out['by_subspace'].items():
        print(f'\n{sname}:')
        for k, s in d.items():
            print(f"  {k:>34s} n={s['n']:4d} med log ratio {s['median_signed_log_ratio']:+.2f} "
                  f"|log ratio| {s['median_abs_log_ratio']:.2f} R2(log) "
                  f"{('%.2f' % s['r2_log']) if s['r2_log'] is not None else '  - '} rho {s['rho']:+.2f}")


if __name__ == '__main__':
    for a in sys.argv[1:]: run(a)
