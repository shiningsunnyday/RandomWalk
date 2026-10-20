"""
Headline metrics for every campaign arm, with the selection rules of PLAN.md §1–2.

  checkpoint / epoch   argmin over epochs of `val_loss_smooth` (identical to `val_loss` when --val_smooth 1),
                       i.e. exactly the quantity the saved checkpoint was chosen by. Validation only.
  configuration        per arm and gas pair, the `fw` setting with the lowest mean validation loss over all
                       split seeds of that arm. Validation only.
  V (headline)         mean over the 6 properties of the across-split population std of test R²
  S                    computed separately by screening.py (needs the checkpoints)
  guards               mean test R², mean test MAE (normalized), mean test Spearman rho, mean test raw-space R²
  arm choice           mean VALIDATION Spearman rho over splits and properties -- PLAN.md's original
                       pre-committed rule. The 2026-10-19 21:15 amendment to validation raw-space R^2 is
                       REVERTED (see PLAN.md): analysis/val_spearman.py recovers the missing Spearman column
                       exactly from the saved checkpoints, so the amendment is unnecessary, and raw-space R^2
                       is not in fact transform-neutral -- it is the training objective of a raw-target run.
                       Both numbers are printed for every arm.

usage: python summarize_campaign.py [arm ...]        -> campaign/outputs/campaign_runs.csv, campaign/RESULTS_table.txt
"""
import os, sys, json, glob, re
import numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__))
PAIRS = {(0, 1): ('H2', 'H2_N2'), (2, 1): ('O2', 'O2_N2'), (4, 3): ('CO2', 'CO2_CH4')}
PROPS = ['H2', 'H2/N2', 'O2', 'O2/N2', 'CO2', 'CO2/CH4']
CFG = ['fw']          # ew, init and direction are fixed per arm; only fw is selected on validation


def val_rho_table():
    """exact validation Spearman per (run dir, property), recovered from the checkpoints by
    analysis/val_spearman.py for runs whose metrics.csv predates the campaign metrics patch"""
    f = os.path.join(H, 'outputs', 'val_spearman.csv')
    if not os.path.exists(f): return {}
    V = pd.read_csv(f)
    return {(r['dir'], r['prop']): (float(r['rho']), float(r['r2_raw'])) for _, r in V.iterrows()}


VR = val_rho_table()


def collect(arms=None):
    runs = []
    for cf in sorted(glob.glob(os.path.join(H, 'outputs', '*', '*', 'predictor_*', 'config.json'))):
        d = os.path.dirname(cf)
        arm = cf.split(os.sep)[-4]
        if arms and arm not in arms: continue
        c = json.load(open(cf)); c = json.loads(c) if isinstance(c, str) else c
        m = os.path.join(d, 'metrics.csv')
        if not os.path.exists(m): continue
        try:
            df = pd.read_csv(m, index_col=0)
        except Exception:
            continue
        if len(df) < c['num_epochs']: continue                      # unfinished run
        sel = 'val_loss_smooth' if 'val_loss_smooth' in df.columns else 'val_loss'
        e = int(df[sel].values.argmin())
        gas, pair = PAIRS[tuple(c['property_cols'])]
        m_t = re.search(r'_t(\d+)', cf.split(os.sep)[-3])
        rec = dict(arm=arm, dir=d, tseed=int(m_t.group(1)) if m_t else 0,
                   pair=pair, seed=c['test_seed'], fw=bool(c['feat_concat_W']),
                   ew=bool(c['edge_weights']), init=c['init'], directed=bool(c.get('ablate_bidir')),
                   logt=bool(c.get('log_target', False)), stratify=c.get('stratify', 'none'),
                   strata=os.path.basename(str(c.get('strata_file') or '')), hold=c.get('holdout_stratum', -1),
                   vsmooth=int(c.get('val_smooth', 1)), huber=float(c.get('huber', 0.0)),
                   wd=float(c.get('weight_decay', 0.0)), hid=c.get('hidden_dim'), drop=c.get('dropout_rate'),
                   epoch=e, val_loss=float(df[sel].iloc[e]), n_epochs=len(df))
        for prop, col in ((gas, f'permeability_{gas}'), (pair.replace('_', '/'), f'selectivity_{pair}')):
            for k in ('mae', 'mse', 'r^2'):
                rec[f'{prop}|{k}'] = float(df[f'heldout_{col}_{k}'].iloc[e])
            for k, src in (('rho', f'heldout_{col}_spearman'), ('valrho', f'{col}_spearman')):
                rec[f'{prop}|{k}'] = float(df[src].iloc[e]) if src in df.columns else np.nan
            if (d, prop) in VR:                      # exact, and available for every arm
                rec[f'{prop}|valrho'] = VR[(d, prop)][0]
            # raw-barrer-space metrics: the `raw_*` columns when the run logged them, otherwise the plain
            # columns, which ARE the raw-space values when --log_target is off (verified identical in the smoke test)
            for k, raw, plain in (('rawr2', f'heldout_raw_{col}_r^2', f'heldout_{col}_r^2'),
                                  ('rawmae', f'heldout_raw_{col}_mae', f'heldout_{col}_mae'),
                                  ('valrawr2', f'raw_{col}_r^2', f'{col}_r^2')):
                src = raw if raw in df.columns else plain
                rec[f'{prop}|{k}'] = float(df[src].iloc[e]) if src in df.columns else np.nan
        runs.append(rec)
    R = pd.DataFrame(runs)
    if len(R):
        # one row per (arm, pair, seed, config): guard against a config having been run twice
        keys = ['arm', 'tseed', 'pair', 'seed', 'fw', 'ew', 'init', 'directed', 'logt', 'stratify', 'strata', 'hold',
                'vsmooth', 'huber', 'wd', 'hid', 'drop', 'n_epochs']
        n0 = len(R)
        R = R.sort_values('dir').drop_duplicates(subset=keys, keep='first')
        if len(R) != n0: print(f'warning: dropped {n0 - len(R)} duplicate runs')
    return R


def select(sub):
    """per gas pair, the fw setting with the lowest mean validation loss over that arm's splits"""
    out = []
    for pair, g in sub.groupby('pair'):
        mm = g.groupby(CFG)['val_loss'].agg(['mean', 'count'])
        k = int(mm['count'].max())
        cand = mm[mm['count'] == k]
        best = cand['mean'].idxmin()
        best = best if isinstance(best, tuple) else (best,)
        sel = g[(g[CFG] == pd.Series(dict(zip(CFG, best)))).all(axis=1)]
        out.append(sel)
    return pd.concat(out) if out else sub


def headline(sel):
    """V and the guards, from the selected runs of one arm"""
    per = {}
    for pair in ('H2_N2', 'O2_N2', 'CO2_CH4'):
        s = sel[sel.pair == pair]
        if not len(s): continue
        for prop in (pair.split('_')[0], pair.replace('_', '/')):
            per[prop] = {k: s[f'{prop}|{k}'].values for k in ('r^2', 'mae', 'rho', 'rawr2', 'rawmae', 'valrho', 'valrawr2')}
    if len(per) < 6:
        return None
    V = float(np.mean([np.nanstd(per[p]['r^2']) for p in PROPS]))
    return dict(
        V=V,
        V_boot=boot_V(per),
        r2=float(np.mean([np.nanmean(per[p]['r^2']) for p in PROPS])),
        r2_sd_of_props=float(np.std([np.nanmean(per[p]['r^2']) for p in PROPS])),
        mae=float(np.mean([np.nanmean(per[p]['mae']) for p in PROPS])),
        rho=float(np.mean([np.nanmean(per[p]['rho']) for p in PROPS])),
        rawr2=float(np.mean([np.nanmean(per[p]['rawr2']) for p in PROPS])),
        rawr2_std=float(np.mean([np.nanstd(per[p]['rawr2']) for p in PROPS])),
        valrho=float(np.mean([np.nanmean(per[p]['valrho']) for p in PROPS])),
        valrawr2=float(np.mean([np.nanmean(per[p]['valrawr2']) for p in PROPS])),
        n_splits=int(max(len(per[p]['r^2']) for p in PROPS)),
        per_prop_r2={p: [float(np.nanmean(per[p]['r^2'])), float(np.nanstd(per[p]['r^2']))] for p in PROPS},
        per_prop_mae={p: [float(np.nanmean(per[p]['mae'])), float(np.nanstd(per[p]['mae']))] for p in PROPS},
        per_prop_rho={p: [float(np.nanmean(per[p]['rho'])), float(np.nanstd(per[p]['rho']))] for p in PROPS},
    )


def boot_V(per, B=2000, seed=0):
    """bootstrap CI for V by resampling split seeds (the same resampled seeds for all 6 properties)"""
    rng = np.random.default_rng(seed)
    n = min(len(per[p]['r^2']) for p in PROPS)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(np.mean([np.nanstd(np.asarray(per[p]['r^2'])[idx]) for p in PROPS]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_vs(selA, selB, col='r^2'):
    """paired per-(pair, seed, property) difference A - B on identical splits; returns mean, sd, t, n"""
    d = []
    for pair in ('H2_N2', 'O2_N2', 'CO2_CH4'):
        # mean over any replicate rows per seed (the `tseed` arm has one row per torch seed)
        a = selA[selA.pair == pair].groupby('seed').mean(numeric_only=True)
        b = selB[selB.pair == pair].groupby('seed').mean(numeric_only=True)
        for s in sorted(set(a.index) & set(b.index)):
            for prop in (pair.split('_')[0], pair.replace('_', '/')):
                va, vb = float(a.loc[s, f'{prop}|{col}']), float(b.loc[s, f'{prop}|{col}'])
                if np.isfinite(va) and np.isfinite(vb): d.append(va - vb)
    d = np.array(d)
    d = d[np.isfinite(d)]
    if len(d) < 2: return None
    return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), n=int(len(d)),
                se=float(d.std(ddof=1) / np.sqrt(len(d))), t=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))))


if __name__ == '__main__':
    R = collect(sys.argv[1:] or None)
    os.makedirs(os.path.join(H, 'outputs'), exist_ok=True)
    R.to_csv(os.path.join(H, 'outputs', 'campaign_runs.csv'), index=False)
    if not len(R):
        print('no completed runs yet'); sys.exit(0)
    print(f'{len(R)} complete runs across {R.arm.nunique()} arms')
    print(R.groupby('arm').size().to_string())
    # an arm enters the ranking only when every one of its jobs has finished, and the hold-out arms and the
    # torch-seed replication arm are listed apart because their "splits" are not the 10 random splits
    def n_jobs(a):
        f = os.path.join(H, 'jobs', f'{a}.txt')
        return sum(1 for _ in open(f)) if os.path.exists(f) else None
    counts = R.groupby('arm').size().to_dict()
    incomplete = {a for a, n in counts.items() if n_jobs(a) is not None and n < n_jobs(a)}
    apart = {a for a in counts if a.startswith('holdout') or a == 'tseed'}
    res, sels = {}, {}
    for arm, g in R.groupby('arm'):
        s = select(g)
        sels[arm] = s
        h = headline(s)
        if h: res[arm] = h
    rank = [a for a in res if a not in incomplete and a not in apart]
    lines = ['Headline metrics per arm   (V = mean over the 6 properties of the across-split std of test R^2; lower is better)',
             'Random-split arms, all runs finished. K = number of split seeds.',
             f'{"arm":16s} {"K":>3s} {"V":>6s} {"V 95% CI":>15s} {"meanR2":>7s} {"MAE":>6s} {"rho":>6s} {"rawR2":>7s} {"valrawR2":>9s} {"valrho":>7s}']
    for arm in sorted(rank, key=lambda a: res[a]['V']):
        h = res[arm]
        lines.append(f'{arm:16s} {h["n_splits"]:3d} {h["V"]:6.3f} [{h["V_boot"][0]:5.3f},{h["V_boot"][1]:5.3f}] '
                     f'{h["r2"]:7.3f} {h["mae"]:6.3f} {h["rho"]:6.3f} {h["rawr2"]:7.3f} {h["valrawr2"]:9.3f} {h["valrho"]:7.3f}')
    other = [a for a in res if a in incomplete or a in apart]
    if other:
        lines += ['', 'NOT comparable in the ranking above (incomplete, or not the 10 random splits):']
        for arm in sorted(other, key=lambda a: res[a]['V']):
            h = res[arm]
            why = 'INCOMPLETE %d/%d runs' % (counts[arm], n_jobs(arm)) if arm in incomplete else (
                'held-out whole stratum' if arm.startswith('holdout') else 'fixed splits x 4 torch seeds')
            lines.append(f'{arm:16s} {h["n_splits"]:3d} {h["V"]:6.3f} [{h["V_boot"][0]:5.3f},{h["V_boot"][1]:5.3f}] '
                         f'{h["r2"]:7.3f} {h["mae"]:6.3f} {h["rho"]:6.3f} {h["rawr2"]:7.3f} {h["valrawr2"]:9.3f} '
                         f'{h["valrho"]:7.3f}   <- {why}')
    if 'base' in sels:
        lines += ['', 'Paired differences against arm `base` on identical splits (test R^2, per pair x seed x property):',
                  f'{"arm":16s} {"mean dR2":>9s} {"se":>7s} {"t":>7s} {"n":>4s}   {"mean drho":>9s} {"se":>7s} {"t":>7s}']
        for arm in sorted(rank):
            if arm == 'base': continue
            p = paired_vs(sels[arm], sels['base'], 'r^2'); q = paired_vs(sels[arm], sels['base'], 'rho')
            if p:
                tail = (f'{q["mean"]:9.3f} {q["se"]:7.3f} {q["t"]:7.2f}' if q
                        else f'{"-":>9s} {"-":>7s} {"-":>7s}')
                lines.append(f'{arm:16s} {p["mean"]:9.3f} {p["se"]:7.3f} {p["t"]:7.2f} {p["n"]:4d}   ' + tail)
        # the same paired test on the raw-barrer-space test R^2, which every arm has and which is comparable
        # across target transforms
        lines += ['', 'Paired differences against `base` in RAW-barrer-space test R^2 (comparable across target spaces):',
                  f'{"arm":16s} {"mean dR2":>9s} {"se":>7s} {"t":>7s} {"n":>4s}']
        for arm in sorted(rank):
            if arm == 'base': continue
            p = paired_vs(sels[arm], sels['base'], 'rawr2')
            if p: lines.append(f'{arm:16s} {p["mean"]:9.3f} {p["se"]:7.3f} {p["t"]:7.2f} {p["n"]:4d}')
    lines += ['', 'Per-property test R^2 (mean +- across-split std) and selected fw:']
    for arm in sorted(rank, key=lambda a: res[a]['V']):
        h = res[arm]
        fw = {p: bool(sels[arm][sels[arm].pair == p].fw.iloc[0]) for p in sorted(sels[arm].pair.unique())}
        lines.append(f'  {arm}  (K={h["n_splits"]}, fw={fw})')
        lines.append('    ' + '  '.join(f'{p} {h["per_prop_r2"][p][0]:.2f}±{h["per_prop_r2"][p][1]:.2f}' for p in PROPS))
        lines.append('    MAE ' + ' '.join(f'{h["per_prop_mae"][p][0]:.2f}±{h["per_prop_mae"][p][1]:.2f}' for p in PROPS))
        lines.append('    rho ' + ' '.join(f'{h["per_prop_rho"][p][0]:.2f}±{h["per_prop_rho"][p][1]:.2f}' for p in PROPS))
    txt = '\n'.join(lines)
    open(os.path.join(H, 'outputs', 'headline.txt'), 'w').write(txt + '\n')
    json.dump({a: res[a] for a in res}, open(os.path.join(H, 'outputs', 'headline.json'), 'w'), indent=1)
    for arm in sels:
        sels[arm].sort_values(['pair', 'seed'])[['arm', 'pair', 'seed', 'fw', 'epoch', 'val_loss', 'dir']].to_csv(
            os.path.join(H, 'outputs', f'selected_{arm}.csv'), index=False)
    print(txt)
