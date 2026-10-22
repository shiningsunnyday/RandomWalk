"""
Collate every baseline row in outputs/results.jsonl with our own model's numbers into one comparison.

Our model's row is read from results_v2/outputs/grid_runs.csv under the same selection rule summarize.py applies
(per gas pair, the configuration with the lowest mean validation loss over the three splits), so the comparison is
computed from the runs rather than copied from the manuscript.

Writes outputs/comparison.md (human-readable) and outputs/comparison.json (for the docx scripts).
usage: python bl_summarize.py
"""
import os, sys, json
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
RV = os.path.dirname(H)
sys.path.insert(0, H)
import bl_data as D

PROPS = ['H2', 'H2/N2', 'O2', 'O2/N2', 'CO2', 'CO2/CH4']
PAIR_OF = {'H2': ('H2', 0), 'H2/N2': ('H2', 1), 'O2': ('O2', 0), 'O2/N2': ('O2', 1),
           'CO2': ('CO2', 0), 'CO2/CH4': ('CO2', 1)}


def ours():
    """MAE / R^2 per property for the validation-selected configuration, straight from the predictor grid."""
    import pandas as pd
    R = pd.read_csv(os.path.join(RV, 'outputs', 'grid_runs.csv'))
    CFG = ['ew', 'fw', 'init', 'directed']
    out = {}
    for pair, props in (('H2_N2', ('H2', 'H2/N2')), ('O2_N2', ('O2', 'O2/N2')), ('CO2_CH4', ('CO2', 'CO2/CH4'))):
        g = R[R.pair == pair]
        m = g.groupby(CFG)['val_loss'].agg(['mean', 'count'])
        best = m[m['count'] == 3]['mean'].idxmin()
        s = g[(g[CFG] == pd.Series(dict(zip(CFG, best)))).all(axis=1)]
        for p in props:
            out[p] = {'mae': (s[f'{p}|mae'].mean(), s[f'{p}|mae'].std(ddof=0)),
                      'r2': (s[f'{p}|r^2'].mean(), s[f'{p}|r^2'].std(ddof=0)),
                      'mae_seeds': s.sort_values('seed')[f'{p}|mae'].tolist(),
                      'r2_seeds': s.sort_values('seed')[f'{p}|r^2'].tolist()}
    return out


def ours_conn():
    """The connection-aware predictor (campaign/conn_predict.py), selected and scored the same way as ours()."""
    import pandas as pd
    p = os.path.join(RV, 'campaign', 'outputs', 'conn', 'conn_grid.csv')
    if not os.path.exists(p):
        return None
    R = pd.read_csv(p)
    out = {}
    for pair, props in (('H2', ('H2', 'H2/N2')), ('O2', ('O2', 'O2/N2')), ('CO2', ('CO2', 'CO2/CH4'))):
        g = R[R.pair == pair]
        m = g.groupby('config')['val_loss'].agg(['mean', 'count'])
        best = m[m['count'] == 3]['mean'].idxmin()
        s = g[g.config == best]
        for q in props:
            out[q] = {'mae': (s[f'{q}|mae'].mean(), s[f'{q}|mae'].std(ddof=0)),
                      'r2': (s[f'{q}|r^2'].mean(), s[f'{q}|r^2'].std(ddof=0)),
                      'mae_seeds': s.sort_values('seed')[f'{q}|mae'].tolist(),
                      'r2_seeds': s.sort_values('seed')[f'{q}|r^2'].tolist()}
    return out


def load():
    rows = {}
    path = os.path.join(H, 'outputs', 'results.jsonl')
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            rows[(r['baseline'], r['pair'])] = r            # a re-run of the same baseline supersedes the earlier row
    by_method = {}
    for (name, pair), r in rows.items():
        for j, prop in enumerate(r['targets']):
            by_method.setdefault(name, {})[prop] = {
                'mae': (r['mae_mean'][j], r['mae_std'][j]), 'r2': (r['r2_mean'][j], r['r2_std'][j]),
                'r2_seeds': [s['r2'][j] for s in r['per_seed']], 'mae_seeds': [s['mae'][j] for s in r['per_seed']],
                'log_target': [s['log_target'] for s in r['per_seed']], 'feat': r['feat'], 'n': r['n']}
    return by_method


def fmt(v, nd=2):
    return f'{v[0]:.{nd}f} ± {v[1]:.{nd}f}'


SI_LABEL = {                                      # table label -> the results.jsonl baseline name
    'Ours (connection-aware)': 'Ours (connection-aware)',
    'Ours (random-walk grammar, published parameterization)': 'Ours (random-walk grammar GNN)',
    'Morgan FP + GP (Tanimoto)': 'Morgan FP + GP (Tanimoto)',
    'Morgan FP + random forest': 'Morgan FP + random forest',
    'Morgan FP + SVR (RBF)': 'Morgan FP + SVR (RBF)',
    'polyBERT + ridge': 'polyBERT + ridge',
    'polyBERT + GP': 'polyBERT + GP (RBF)',
    'ChemBERTa + ridge': 'ChemBERTa + ridge',
    'Chemprop D-MPNN': 'Chemprop D-MPNN',
    'Bag-of-Motifs (+expert)': 'Bag-of-Motifs (+expert)',
    'Bag-of-Motifs': 'Bag-of-Motifs',
    'HM-GNN (+expert)': 'HM-GNN (+expert)',
    'HM-GNN': 'HM-GNN',
    'RDKit descriptors + XGBoost': 'RDKit descriptors + XGBoost',
    'RDKit descriptors + ridge': 'RDKit descriptors + ridge',
    'Group-contribution regression': 'GC regression (motif counts)',
    'Predict the training mean': 'Training mean (reference)',
}


def si_tables(M, mean_mae, med_r2, min_r2):
    """Two SI tables: the tuned baselines, and the same baselines restricted to the untransformed target (the target
    our own model uses). Ours is always the first row; the rest are ordered by mean MAE within each table."""
    props_hdr = ['Method'] + ['H₂', 'H₂/N₂', 'O₂', 'O₂/N₂', 'CO₂', 'CO₂/CH₄']
    out = {}
    for key, suffix in (('tuned', ''), ('raw', ' [raw target]')):
        rows, notes = [], []
        avail = [(lab, nm + suffix if nm != SI_LABEL['Ours (random-walk grammar)'] else nm)
                 for lab, nm in SI_LABEL.items()]
        avail = [(lab, nm) for lab, nm in avail if nm in M and all(p in M[nm] for p in PROPS)]
        ours = [(lab, nm) for lab, nm in avail if nm.startswith('Ours')]
        rest = sorted([(lab, nm) for lab, nm in avail if not nm.startswith('Ours')], key=lambda x: mean_mae[x[1]])
        for lab, nm in ours + rest:
            flag = '*' if min_r2[nm] < -1 else ''
            # compact, because a cell has to hold two numbers in a seventh of the text width; a typographic minus
            # also stops a negative R2 from looking like a line-break hyphen
            def cell(p):
                r2 = M[nm][p]['r2'][0]
                # a diverged fit's R2 can run to four digits, which no cell this narrow can hold, and the exact
                # value carries nothing beyond "diverged"; the caption defines the symbol.
                r2s = '≪0' if r2 <= -10 else (f'{r2:.1f}' if r2 <= -1 else f'{r2:.2f}')
                return f"{M[nm][p]['mae'][0]:.2f}/{r2s}".replace('-', '\u2212')
            rows.append([lab + flag] + [cell(p) for p in PROPS])
            if flag:
                notes.append(lab)
        out[key] = {'header': props_hdr, 'rows': rows, 'degenerate': notes,
                    'missing': [lab for lab, nm in SI_LABEL.items()
                                if (nm + suffix if not nm.startswith('Ours') else nm) not in M]}
    json.dump(out, open(os.path.join(H, 'outputs', 'si_tables.json'), 'w'), indent=1)
    print('\nSI tables written:', {k: len(v['rows']) for k, v in out.items()},
          '| not yet available:', out['tuned']['missing'])


def main():
    M = load()
    M['Ours (random-walk grammar GNN)'] = ours()
    _conn = ours_conn()
    if _conn:
        M['Ours (connection-aware)'] = _conn
    order = (['Ours (connection-aware)'] if _conn else []) + ['Ours (random-walk grammar GNN)'] + \
        sorted(k for k in M if not k.startswith('Ours'))
    # headline: mean over the six properties, which is how the ablation table already summarizes a configuration
    mean_mae = {k: np.mean([M[k][p]['mae'][0] for p in PROPS if p in M[k]]) for k in order}
    mean_r2 = {k: np.mean([M[k][p]['r2'][0] for p in PROPS if p in M[k]]) for k in order}
    complete = [k for k in order if all(p in M[k] for p in PROPS)]

    # R^2 is not robust here: a model fitted in log space can extrapolate far above the training maximum on one
    # high-permeability test polymer, and exponentiating that error gives an R^2 of -30 on that single split while
    # MAE barely moves. So rank by MAE, and show the median and the worst single (property, seed) R^2 alongside the
    # mean so a degenerate row is visible instead of silently dragging an average down.
    all_r2 = {k: [v for p in PROPS if p in M[k] for v in M[k][p].get('r2_seeds', [])] for k in order}
    med_r2 = {k: float(np.median(all_r2[k])) if all_r2[k] else float('nan') for k in order}
    min_r2 = {k: float(np.min(all_r2[k])) if all_r2[k] else float('nan') for k in order}

    L = ['# Prediction baselines under the v2 protocol', '',
         'Same three train/test splits, same validation-only model selection, same training-split normalization as',
         'Table 2. MAE and R² are on the normalized targets, mean ± population standard deviation over the three',
         'splits. `log` marks how many of the 18 (property, split) fits selected the log-target option on validation',
         '(our own model is not offered that option, so its cell is blank); a `[raw target]` row is the same baseline',
         'with that option removed.', '',
         'Rows are ranked by mean MAE, not mean R². A model fitted on log-permeability can extrapolate above the',
         'training maximum on a single high-permeability test polymer, and exponentiating that error produces an R²',
         'of −30 on that one split while MAE hardly changes; the mean R² of such a row is meaningless. `median R²`',
         'and `worst R²` are over all 18 (property, split) fits and make those rows visible.', '',
         '## Averaged over the six properties', '',
         '| Method | mean MAE ↓ | mean R² ↑ | median R² | worst R² | log |', '|---|---|---|---|---|---|']
    for k in sorted(complete, key=lambda k: mean_mae[k]):
        lg = [b for p in PROPS for b in M[k][p].get('log_target', [])]
        flag = ' ⚠' if min_r2[k] < -1 else ''
        L.append(f'| {k}{flag} | {mean_mae[k]:.3f} | {mean_r2[k]:.3f} | {med_r2[k]:.3f} | {min_r2[k]:.2f} | '
                 f'{"" if not lg else f"{sum(lg)}/{len(lg)}"} |')
    L += ['', '⚠ = at least one (property, split) fit with R² < −1, i.e. a degenerate extrapolation; read MAE for '
              'these rows.']
    incomplete = [k for k in order if k not in complete]
    if incomplete:
        L += ['', f'Not yet complete for all six properties: {", ".join(incomplete)}.']

    for metric, nd, arrow in (('mae', 2, '↓'), ('r2', 2, '↑')):
        L += ['', f'## {"MAE" if metric == "mae" else "R²"} {arrow} per property', '',
              '| Method | ' + ' | '.join(PROPS) + ' |', '|---' * (len(PROPS) + 1) + '|']
        for k in order:
            cells = [fmt(M[k][p][metric], nd) if p in M[k] else '—' for p in PROPS]
            L.append(f'| {k} | ' + ' | '.join(cells) + ' |')

    open(os.path.join(H, 'outputs', 'comparison.md'), 'w').write('\n'.join(L) + '\n')
    si_tables(M, mean_mae, med_r2, min_r2)
    json.dump({'props': PROPS, 'methods': order, 'mean_mae': mean_mae, 'mean_r2': mean_r2,
               'complete': complete,
               'table': {k: {p: {m: list(M[k][p][m]) for m in ('mae', 'r2')} for p in M[k]} for k in order}},
              open(os.path.join(H, 'outputs', 'comparison.json'), 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
