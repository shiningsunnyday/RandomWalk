"""
Quantitative figures for the revision, from results_v2 outputs.

  Figs S2–S4  predicted (ensemble mean ± std) performance of the 1000 generated structures whose ensemble std is below
              the caption thresholds, the measured values of the 114 dataset polymers, the ensemble's predictions for
              those dataset polymers, the Pareto front and the Robeson 2008 upper bound (as plot_group_contrib_with_err).
  Fig 5       measured (filled; triplicate std, SI Tables S3/S5) vs predicted (open; ensemble std) permeability and
              selectivity of the four synthesized PIM-PIs for CO2/CH4, O2/N2, H2/N2, with the 2008 upper bound.

usage: python results_v2/plots.py      -> results_v2/outputs/figs/
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(H, 'outputs'); F = os.path.join(O, 'figs'); os.makedirs(F, exist_ok=True)
import gen_set as GEN                       # which generated set the screening figures show
import pred_set as PRED                     # which trained predictor supplies the predictions
GAS = {'H2': ('H2/N2', 9.765e4, -1.4841, (500, 10)), 'O2': ('O2/N2', 1.396e6, -5.666, (100, 2)),
       'CO2': ('CO2/CH4', 5.369e6, -2.636, (1000, 5))}
COL = {'H2': 0, 'N2': 1, 'O2': 2, 'CH4': 3, 'CO2': 4}
tex = lambda s: s.replace('H2', 'H$_2$').replace('N2', 'N$_2$').replace('O2', 'O$_2$').replace('CH4', 'CH$_4$').replace('CO2', 'CO$_2$')


def plain_ticks(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator
    f = FuncFormatter(lambda v, _: f'{v:g}')
    ax.yaxis.set_major_formatter(f); ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f'{v:g}' if str(f'{v:g}')[0] in '2345' else ''))


def load_ens(p):
    d = {}
    for l in open(p):
        t = l.split(); d[(t[0], t[1])] = (float(t[2]), float(t[4]))
    return d


def pareto(x, y):
    """non-dominated points when maximizing both x and y"""
    idx = np.argsort(-x); best = -np.inf; on = np.zeros(len(x), bool)
    for i in idx:
        if y[i] > best:
            on[i] = True; best = y[i]
    return on


def upper_bound(ax, k, n, xlim):
    xs = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
    ax.plot(xs, (xs / k) ** (1 / n), c='tab:blue' if ax.get_label() == 'screen' else '0.7', lw=1.5, zorder=0)


def screening_figures():
    gen = load_ens(os.path.join(H, PRED.S9))
    data = load_ens(os.path.join(H, PRED.DATASET))
    walks_gen = [l.split()[0] for l in open(os.path.join(O, GEN.WALKS))]
    rows = [l.split() for l in open(os.path.join(H, 'data/group-contrib/polymer_walks_v2_preprocess.txt'))]
    counts = {}
    for gas, (pair, k, n, (tp, ts)) in GAS.items():
        a, b = COL[gas], COL[pair.split('/')[1]]
        gt, pred = [], []
        for r in rows:
            v = r[-1].strip('()').split(',')
            if v[a] in '-_' or v[b] in '-_': continue
            gt.append((float(v[a]), float(v[a]) / float(v[b])))
            plain = __import__('re').sub(r'-\d+>', '->', r[1])
            pred.append((data[(plain, gas)][0], data[(plain, pair)][0]))
        gt, pred = np.array(gt), np.array(pred)
        P = np.array([gen[(w, gas)] for w in walks_gen]); S = np.array([gen[(w, pair)] for w in walks_gen])
        keep = (P[:, 1] < tp) & (S[:, 1] < ts) & (P[:, 0] > 0) & (S[:, 0] > 0)
        counts[gas] = int(keep.sum())
        P, S = P[keep], S[keep]
        on = pareto(np.r_[P[:, 0], gt[:, 0]], np.r_[S[:, 0], gt[:, 1]])
        on_gen, on_gt = on[:len(P)], on[len(P):]
        fig, ax = plt.subplots(figsize=(6.2, 4.8)); ax.set_label('screen')
        ax.scatter(gt[~on_gt, 0], gt[~on_gt, 1], c='green', label='ground-truth values of original monomers')
        ax.scatter(gt[on_gt, 0], gt[on_gt, 1], c='green', marker='v', label='original monomers on the pareto front')
        ax.scatter(pred[:, 0], pred[:, 1], c='red', label='predicted values of original monomers')
        ax.errorbar(P[~on_gen, 0], S[~on_gen, 0], xerr=P[~on_gen, 1], yerr=S[~on_gen, 1], c='b', fmt='o',
                    label='predicted values of novel monomers\n(with std error bar over 3 splits)')
        ax.errorbar(P[on_gen, 0], S[on_gen, 0], xerr=P[on_gen, 1], yerr=S[on_gen, 1], c='b', fmt='v',
                    label='novel monomers on the pareto front')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, which='major'); plain_ticks(ax)
        allx = np.r_[gt[:, 0], pred[:, 0], P[:, 0]]
        allx = allx[allx > 0]  # a few dataset polymers get negative predicted means; not drawable on a log axis
        upper_bound(ax, k, n, (allx.min() * 0.8, allx.max() * 1.2))
        ax.set_xlabel(tex(f'{gas} permeability'), fontsize=16); ax.set_ylabel(tex(f'{pair} selectivity'), fontsize=16)
        ax.legend(fontsize=8, loc='upper right'); fig.tight_layout()
        fig.savefig(os.path.join(F, f'screening_{gas}.png'), dpi=220); plt.close(fig)
    json.dump(counts, open(os.path.join(F, 'screening_counts.json'), 'w'))
    print('structures within error thresholds:', counts)


EXP = {  # SI Tables S3/S5 (triplicate mean ± std); unchanged from the submission
    'PIM-PI-1': dict(H2=(470, 50), O2=(110, 10), CO2=(660, 70), **{'H2/N2': (16.1, 2.1), 'O2/N2': (3.7, 0.4), 'CO2/CH4': (15.8, 3.1)}),
    'PIM-PI-2': dict(H2=(270, 4), O2=(51.6, 0.2), CO2=(313, 4), **{'H2/N2': (21.4, 0.4), 'O2/N2': (4.1, 0.1), 'CO2/CH4': (18.6, 0.3)}),
    'PIM-PI-3': dict(H2=(400, 20), O2=(85, 6), CO2=(490, 30), **{'H2/N2': (18.4, 2.0), 'O2/N2': (3.9, 0.5), 'CO2/CH4': (17.3, 1.5)}),
    'PIM-PI-4': dict(H2=(220, 10), O2=(40, 2), CO2=(227, 2), **{'H2/N2': (22.8, 2.4), 'O2/N2': (4.1, 0.4), 'CO2/CH4': (18.3, 2.3)}),
}
COLORS = {'PIM-PI-1': '#c00000', 'PIM-PI-2': '#1f4e9c', 'PIM-PI-3': '#2e7d4f', 'PIM-PI-4': '#7b3fb0'}


def fig5():
    pred = load_ens(os.path.join(H, PRED.FIG5))
    walk_of = json.load(open(os.path.join(H, 'inputs', 'experimental_walks.json')))
    # Permeabilities are shown on our own lab's scale: the predictions are divided by the literature-vs-lab factor
    # identified on PIM-PI-4, the one repeat unit measured in both protocols (compute_numbers.py -> calibration).
    # That correction is part of the method, so the figure shows the corrected values and only those; selectivities
    # are NOT calibrated, because a ratio of two permeabilities on the same film already cancels the offset.
    cal = json.load(open(os.path.join(O, 'numbers.json')))['calibration']['factors']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, gas, lab in zip(axes, ('CO2', 'O2', 'H2'), 'ABC'):
        pair, k, n, _ = GAS[gas]
        xs = []
        for p, w in walk_of.items():
            (ex, exs), (ey, eys) = EXP[p][gas], EXP[p][pair]
            (rx, rxs), (py, pys) = pred[(w, gas)], pred[(w, pair)]
            px, pxs = rx / cal[gas], rxs / cal[gas]
            c = COLORS[p]
            ax.plot([ex, px], [ey, py], c=c, lw=0.8, ls=':', zorder=1)
            ax.errorbar(ex, ey, xerr=exs, yerr=eys, fmt='o', c=c, ms=7, capsize=2, zorder=3)
            ax.errorbar(px, py, xerr=pxs, yerr=pys, fmt='D', c=c, mfc='white', ms=7, capsize=2, zorder=2)
            xs += [ex, max(px - pxs, ex * 0.5), px + pxs]
        ax.set_xscale('log'); ax.set_yscale('log')
        plain_ticks(ax)
        lo, hi = 10 ** np.floor(np.log10(min(xs))), 10 ** np.ceil(np.log10(max(xs)))
        ax.set_xlim(lo, hi)
        xx = np.logspace(np.log10(lo), np.log10(hi), 100); ax.plot(xx, (xx / k) ** (1 / n), c='0.7', lw=1.5, zorder=0)
        ax.text(0.97, 0.95, '2008', transform=ax.transAxes, ha='right', va='top', color='0.5', fontsize=9)
        ax.set_xlabel(tex(f'{gas} permeability (barrer)'), fontsize=12); ax.set_ylabel(tex(f'{pair} selectivity'), fontsize=12)
        ax.text(0.04, 0.95, lab, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', ls='', c=c, label=p) for p, c in COLORS.items()]
    handles += [Line2D([], [], marker='o', ls='', c='k', label='experiment'),
                Line2D([], [], marker='D', ls='', c='k', mfc='white', label='prediction')]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(F, 'fig5.png'), dpi=220); plt.close(fig)


if __name__ == '__main__':
    screening_figures()
    fig5()
