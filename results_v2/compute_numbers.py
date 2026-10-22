"""
Every number written into the main text, computed from results_v2 outputs -> outputs/numbers.json.

  table1   Valid (Walk) / (SMARTS) / (SMILES) from the seeded random-sampling counters (sample_walks);
           Unique / Novel / Diversity / Retro* on the best-first top 1000, using clean SMILES (unconsumed R stand-ins
           dropped, H-capped, no atom maps): Unique = distinct canonical SMILES, Novel = not among the 114 dataset
           SMILES, Diversity = 1 - mean pairwise Tanimoto (Morgan r=3, 2048 bits), Retro* = success rate (retro.py).
  table2   test MAE and R2 of the validation-selected models (summarize.py), Table 2 row order.
  motif statistics, generated-walk length, experiment-vs-prediction ranges, ablation sentence.
usage: python results_v2/compute_numbers.py
"""
import os, re, json, math
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')
H = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(H, 'outputs'); D = os.path.join(H, 'data', 'group-contrib')
import gen_set as GEN                       # which generated set Table 1 reports
import pred_set as PRED                     # which trained predictor Table 2 and the ensembles come from
N = {}

# ---- Table 1
c = json.load(open(os.path.join(O, 'gen', 'random_n1000_s0_counters.json')))
t, wi, ci = c['num_tried'], c['walk_invalid'], c['chemical_invalid']
valid_walk, valid_smarts, valid_smiles = 1 - wi / t, 1 - ci / (t - wi), c['num_valid'] / (t - wi - ci)
gen = [l.rstrip('\n').split(' ') for l in open(os.path.join(O, GEN.CLEAN))]
assert all(g[1] == '1' for g in gen), 'some best-first structures do not sanitize'
can = [Chem.MolToSmiles(Chem.MolFromSmiles(g[0])) for g in gen]
train = {Chem.MolToSmiles(Chem.MolFromSmiles(l.split()[0])) for l in open(os.path.join(D, 'polymer_walks_v2_smiles.txt')) if l.strip()}
unique = len(set(can)) / len(can); novel = sum(x not in train for x in can) / len(can)
fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 3, 2048) for x in can]
sim = sum(sum(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])) for i in range(len(fps)))
diversity = 1 - sim / (len(fps) * (len(fps) - 1) / 2)
retro = [l.split()[-1] for l in open(os.path.join(O, GEN.RETRO)) if l.strip()]
rs = sum(x == 'True' for x in retro) / len(retro)
pct = lambda x: f'{100 * x:.0f}%'
N['table1'] = [pct(valid_walk), pct(valid_smarts), pct(valid_smiles), pct(unique), pct(novel), f'{diversity:.2f}', pct(rs)]
N['table1_exact'] = dict(valid_walk=valid_walk, valid_smarts=valid_smarts, valid_smiles=valid_smiles, unique=unique,
                         novel=novel, diversity=diversity, retro=rs, counters=c, n_gen=len(gen))

# ---- where sampling fails (reviewer asked for failure cases, and Table 1 only reports the stages after closure).
# A repeat unit requires the walk to return to the motif it started from; sample_walk only sets good=True in that
# case, so every trajectory that runs out of probability mass, revisits itself or cannot be extended is discarded
# before any validity check. That rejection, not chemical validity, is the dominant failure mode.
ns, nt = c['num_sampled'], c['num_tried']
N['failure_modes'] = {
    'rows': [
        ['Sampled trajectories', 'did not close back to the starting motif', f'{ns - nt:,}', f'{100 * (ns - nt) / ns:.1f}%'],
        ['Closed walks', 'no legal R-group attachment (invalid walk)', f"{c['walk_invalid']}", f"{100 * c['walk_invalid'] / nt:.1f}%"],
        ['Assembled walks', 'motif-level assembly failed', f"{c['chemical_invalid']}", f"{100 * c['chemical_invalid'] / (nt - c['walk_invalid']):.1f}%"],
        ['Assembled walks', 'final structure failed sanitization', f"{c['chemical_invalid_smiles']}",
         f"{100 * c['chemical_invalid_smiles'] / (nt - c['walk_invalid'] - c['chemical_invalid']):.1f}%"],
        ['Valid structures', 'kept', f"{c['num_valid']:,}", f'{100 * c["num_valid"] / ns:.1f}%'],
    ],
    'header': ['Stage', 'Outcome', 'Count', 'Share'],
    'closure_pct': f'{100 * nt / ns:.1f}', 'n_sampled': f'{ns:,}', 'n_closed': f'{nt:,}'}

# ---- Table 2 (row order: H2, H2/N2, O2, O2/N2, CO2, CO2/CH4)
R = pd.read_csv(os.path.join(H, PRED.GRID))
CFG = ['config'] if PRED.STEM == 'conn' else ['ew', 'fw', 'init', 'directed']
def select(sub):
    out = []
    for pair, g in sub.groupby('pair'):
        m = g.groupby(CFG)['val_loss'].agg(['mean', 'count'])
        best = m[m['count'] == 3]['mean'].idxmin()
        if not isinstance(best, tuple):
            best = (best,)
        out.append(g[(g[CFG] == pd.Series(dict(zip(CFG, best)))).all(axis=1)])
    return pd.concat(out)
sel = select(R)
rows = []
for pair, props in (('H2_N2', ('H2', 'H2/N2')), ('O2_N2', ('O2', 'O2/N2')), ('CO2_CH4', ('CO2', 'CO2/CH4'))):
    s = sel[sel.pair == (pair.split('_')[0] if PRED.STEM == 'conn' else pair)]
    assert len(s) == 3, (pair, len(s))
    for p in props:
        mae, r2 = s[f'{p}|mae'].values, s[f'{p}|r^2'].values
        rows.append([f'{mae.mean():.2f} ± {mae.std():.2f}', f'{r2.mean():.2f} ± {r2.std():.2f}'])
N['table2'] = rows
N['selected_configs'] = {pair: dict(sel[sel.pair == pair][CFG].iloc[0].astype(str)) for pair in sel.pair.unique()}
N['table2_model'] = PRED.STEM
# the published grid is still read for the ablation sentence about edge weights and directedness, which describes
# the inherited architecture rather than the model Table 2 now reports
_PUB = pd.read_csv(os.path.join(O, 'grid_runs.csv'))
_PCFG = ['ew', 'fw', 'init', 'directed']
_psel = []
for _pair, _g in _PUB.groupby('pair'):
    _m = _g.groupby(_PCFG)['val_loss'].agg(['mean', 'count'])
    _best = _m[_m['count'] == 3]['mean'].idxmin()
    _psel.append(_g[(_g[_PCFG] == pd.Series(dict(zip(_PCFG, _best)))).all(axis=1)])
_psel = pd.concat(_psel)
N['published_selected_configs'] = {p: dict(_psel[_psel.pair == p][_PCFG].iloc[0].astype(str))
                                   for p in _psel.pair.unique()}

# ---- design-space statistics
names = open(os.path.join(D, 'all_groups', 'names.txt')).read().split()
heavy = [Chem.MolFromMolFile(os.path.join(D, 'all_groups', f'{i}.mol'), sanitize=False).GetNumHeavyAtoms() for i in range(1, len(names) + 1)]
N['n_motifs'] = len(names); N['motif_size'] = f'{np.mean(heavy):.1f}'
steps = [len(re.findall(r'[PSL]\d+', g[2])) - 1 for g in gen]
N['gen_motifs_mean'] = float(np.mean(steps)); N['gen_motifs_max'] = int(max(steps))
N['steps_text'] = f'{np.mean(steps):.1f} (at most {max(steps)})'
data_steps = [len(re.findall(r'[PSL]\d+', l.split()[1])) - 1 for l in open(os.path.join(D, 'polymer_walks_v2_preprocess.txt'))]
N['dataset_motifs_mean'] = float(np.mean(data_steps))

# ---- experiment vs prediction (synthesized polymers)
import sys; sys.path.insert(0, H)
from plots import EXP, load_ens
pred = load_ens(os.path.join(H, PRED.FIG5)); walk_of = json.load(open(os.path.join(H, 'inputs', 'experimental_walks.json')))
ratio = lambda prop: [pred[(w, prop)][0] / EXP[p][prop][0] for p, w in walk_of.items()]
h2 = ratio('H2'); N['h2_ratio_range'] = f'{min(h2):.1f} – {max(h2):.1f}'
N['co2ch4_within_pct'] = int(math.ceil(100 * max(abs(r - 1) for r in ratio('CO2/CH4'))))
N['h2n2_within_pct'] = int(math.ceil(100 * max(abs(r - 1) for r in ratio('H2/N2'))))
N['perm_ratios'] = {g: ratio(g) for g in ('H2', 'O2', 'CO2')}
N['sel_ratios'] = {g: ratio(g) for g in ('H2/N2', 'O2/N2', 'CO2/CH4')}

# ---- literature vs in-house measurement, for the one polymer measured both ways.
# Our synthesized PIM-PI-4 is the durene polyimide, and the dataset already contains it under the name 'PIM-PI-1'
# with a different (equivalent) motif decomposition: the dataset walk builds the spirobisindane-dioxin-bisimide core
# out of L16/S32/L3, ours out of the single motif L5. So its label is a literature permeability for the same repeat
# unit we measured ourselves, which gives a direct estimate of the label-vs-lab offset the model inherits.
_andlist = lambda xs: ', '.join(xs[:-1]) + ' and ' + xs[-1]
LIT_ROW, LAB_POLYMER = 'PIM-PI-1', 'PIM-PI-4'
_lit = next(l for l in open(os.path.join(D, 'polymer_walks_v2_preprocess.txt')) if l.split()[0] == LIT_ROW)
_cols = dict(zip(('H2', 'N2', 'O2', 'CH4', 'CO2'), [float(x) for x in _lit.split()[-1].strip('(').rstrip(')').split(',')]))
assert 'L16' in _lit and 'S20' in _lit, _lit                      # durene bridge on the L16-based core
_lab = {g: EXP[LAB_POLYMER][g][0] for g in ('H2', 'O2', 'CO2')}
_lvl = {g: _cols[g] / _lab[g] for g in _lab}
N['lit_vs_lab'] = {'dataset_row': LIT_ROW, 'measured_as': LAB_POLYMER,
                   'lit': {g: _cols[g] for g in _lab}, 'lab': _lab, 'ratios': _lvl,
                   'lit_text': _andlist([f'{_cols[g]:.0f}' for g in ('H2', 'O2', 'CO2')]),
                   'lab_text': _andlist([f'{_lab[g]:.0f}' for g in ('H2', 'O2', 'CO2')]),
                   'ratio_range': f'{min(_lvl.values()):.1f} – {max(_lvl.values()):.1f}'}

# ---- calibrating the predicted permeabilities on the one polymer measured in both protocols.
# The offset above is a property of the measurement, not of the model: a literature permeability and ours for the
# same repeat unit differ by 2.4-4.8x. Dividing the predicted permeabilities by that factor puts them on our own
# lab's scale, which is the scale the other three polymers were measured on. The factor is identified on the
# reference polymer alone, so the three novel polymers are a genuine out-of-sample test of it. Selectivities are
# NOT calibrated: a selectivity is a ratio of two permeabilities measured on the same film, so the offset cancels
# before any correction and applying one would double-count it (and does, measurably: 4/9 within error becomes 1/9).
_CPERM = ('H2', 'O2', 'CO2')
_c = {g: _lvl[g] for g in _CPERM}                      # literature / our lab on the reference, one factor per gas
_novel = [p for p in walk_of if p != LAB_POLYMER]
def _cal_stats(group, c):
    rs = [(pred[(walk_of[n], g)][0] / c[g]) / EXP[n][g][0] for n in group for g in _CPERM]
    within = sum(abs(pred[(walk_of[n], g)][0] / c[g] - EXP[n][g][0]) <= EXP[n][g][1] + pred[(walk_of[n], g)][1] / c[g]
                 for n in group for g in _CPERM)
    return {'n': len(rs), 'geomean': float(np.exp(np.mean(np.log(rs)))),
            'mean_abs_log10': float(np.mean(np.abs(np.log10(rs)))), 'within': int(within),
            'range': [float(min(rs)), float(max(rs))]}
_one = float(np.exp(np.mean(np.log([pred[(walk_of[LAB_POLYMER], g)][0] / EXP[LAB_POLYMER][g][0] for g in _CPERM]))))
N['calibration'] = {
    'factors': _c, 'reference': LAB_POLYMER, 'novel': _novel,
    'factor_text': ', '.join(f'{_c[g]:.1f}x for {g}' for g in _CPERM),
    'raw': _cal_stats(_novel, {g: 1.0 for g in _CPERM}), 'cal': _cal_stats(_novel, _c),
    'per_gas': {g: {'c': _c[g],
                    'geomean_raw': float(np.exp(np.mean(np.log([pred[(walk_of[n], g)][0] / EXP[n][g][0]
                                                                for n in _novel])))),
                    'geomean_cal': float(np.exp(np.mean(np.log([(pred[(walk_of[n], g)][0] / _c[g]) / EXP[n][g][0]
                                                                for n in _novel])))),
                    'within_cal': int(sum(abs(pred[(walk_of[n], g)][0] / _c[g] - EXP[n][g][0])
                                          <= EXP[n][g][1] + pred[(walk_of[n], g)][1] / _c[g] for n in _novel))}
                for g in _CPERM},
    # a one-parameter alternative: a single gain, the geometric mean of predicted/measured over the reference's three
    # gases. Reported as a sensitivity check, because three factors identified from one polymer invites the question
    # of whether the conclusion depends on that choice. It does not: the scatter falls either way.
    'single_gain': {'c': _one, **_cal_stats(_novel, {g: _one for g in _CPERM})}}
N['calibration']['cal']['improvement_pct'] = int(round(100 * (1 - N['calibration']['cal']['mean_abs_log10']
                                                             / N['calibration']['raw']['mean_abs_log10'])))
# how closely the corrected permeabilities match, as a factor a reader can quote: the worst deviation in either
# direction, over all four polymers and all three gases, and over the three that are not the reference
def _worst(group):
    rs = [(pred[(walk_of[n], g)][0] / _c[g]) / EXP[n][g][0] for n in group for g in _CPERM]
    return float(max(max(r, 1 / r) for r in rs))
N['calibration']['within_factor_all'] = _worst(list(walk_of))
N['calibration']['within_factor_novel'] = _worst(_novel)
N['calibration']['within_text_all'] = f"{N['calibration']['within_factor_all']:.1f}"
N['calibration']['within_text_novel'] = f"{N['calibration']['within_factor_novel']:.1f}"

# ---- the connection-aware predictor (campaign/conn_gnn.py), which is the model this revision prescribes.
# Metrics are on the back-transformed raw scale throughout, so every arm and every baseline is on one footing;
# `base` is the published configuration run under the same ten-split protocol.
_CONN = os.path.join(H, 'campaign', 'outputs', 'conn')
if os.path.isdir(_CONN):
    import csv as _csv

    def _cells(arm):
        p = os.path.join(_CONN, f'{arm}.csv')
        if not os.path.exists(p):
            return {}
        return {(r['prop'], int(r['seed'])): (float(r.get('raw_mae') or r['mae']), float(r.get('raw_r2') or r['r2']))
                for r in _csv.DictReader(open(p))}

    _cr = pd.read_csv(os.path.join(H, 'campaign', 'outputs', 'campaign_runs.csv'))
    _P6 = ['H2', 'H2/N2', 'O2', 'O2/N2', 'CO2', 'CO2/CH4']
    _b = {}
    _g = _cr[_cr.arm == 'base']
    for _p in _P6:
        for _s, _gs in _g[_g[f'{_p}|r^2'].notna()].groupby('seed'):
            _x = _gs.loc[_gs['val_loss'].idxmin()]
            _b[(_p, _s)] = (_x[f'{_p}|rawmae'], _x[f'{_p}|rawr2'])

    def _stat(c, ref=None):
        out = {'mae': float(np.mean([v[0] for v in c.values()])),
               'r2': float(np.mean([v[1] for v in c.values()])),
               'n': len(c)}
        if ref:
            k = sorted(set(c) & set(ref))
            d_ = np.array([c[i][0] - ref[i][0] for i in k])
            out['dmae'] = float(d_.mean())
            out['t'] = float(d_.mean() / (d_.std(ddof=1) / len(d_) ** 0.5))
        return out

    N['conn'] = {'base': _stat(_b)}
    for _a, _tag in (('conn_r', 'conn'), ('conn_nored', 'nored'), ('conn_redall', 'redall'),
                     ('conn_fp', 'fp'), ('conn_fixed', 'fixed')):
        _c = _cells(_a)
        if _c:
            N['conn'][_tag] = _stat(_c, _b)
    _cn, _nr = _cells('conn_r'), _cells('conn_nored')
    if _cn and _nr:
        N['conn']['conn_vs_nored'] = _stat(_cn, _nr)

# ---- molecular weight vs measured permeability (reviewer asked whether the films are representative).
# The GPC values are reported per polymer in the synthesis section but never related to the transport data.
from scipy.stats import spearmanr
_g = {k: v for k, v in json.load(open(os.path.join(H, 'inputs', 'gpc.json'))).items() if not k.startswith('_')}
_order = sorted(_g)
_mn = [_g[p]['Mn_kDa'] for p in _order]
N['gpc'] = {'mn': {p: _g[p]['Mn_kDa'] for p in _order}, 'disp': {p: _g[p]['D'] for p in _order},
            'bet': {p: _g[p]['BET_m2_g'] for p in _order},
            'mn_range': f'{min(_mn):.1f} – {max(_mn):.1f}',
            'mn_low_text': ', '.join(f'{_g[p]["Mn_kDa"]:.1f}' for p in _order if _g[p]['Mn_kDa'] < 10),
            'disp_range': f'{min(v["D"] for v in _g.values()):.2f} – {max(v["D"] for v in _g.values()):.2f}',
            'bet_range': f'{min(v["BET_m2_g"] for v in _g.values())} – {max(v["BET_m2_g"] for v in _g.values())}'}
for gas in ('H2', 'O2', 'CO2'):
    rho, pv = spearmanr(_mn, [EXP[p][gas][0] for p in _order])
    N['gpc'][f'spearman_mn_{gas}'] = [round(float(rho), 2), round(float(pv), 3)]

# ---- ablation sentence (from summarize.py's ablation.json)
ab = {(r[0].split(' — ')[0], r[1]): (r[2], r[3])                 # the default row's label carries a ' — default' tag
      for r in json.load(open(os.path.join(O, 'ablation.json')))['rows']}
m = lambda k: float(ab[k][0].split(' ±')[0]); q = lambda k: float(ab[k][1].split(' ±')[0])
u0, d0, u1, d1 = ('undirected', 'no'), ('directed (walk order)', 'no'), ('undirected', 'yes'), ('directed (walk order)', 'yes')
dir_better = m(d0) <= m(u0) and q(d0) >= q(u0); ew_worse = m(u1) > m(u0) and m(d1) > m(d0)
assert all(v['directed'] == 'True' and v['ew'] == 'False'
           for v in N['published_selected_configs'].values()), N['published_selected_configs']
N['ablation_text'] = (
    f"{'directed message passing improved' if dir_better else 'directed message passing did not improve'} the test "
    f"metrics (MAE {m(d0):.3f} vs {m(u0):.3f}, R² {q(d0):.2f} vs {q(u0):.2f} without edge weights), "
    f"{'whereas edge weights worsened them' if ew_worse else 'and edge weights changed them little'} "
    f"(MAE {m(u1):.3f} undirected and {m(d1):.3f} directed), and the configuration selected on the validation set "
    f"uses directed, unweighted message passing for all three gas pairs.")
# The directed-vs-undirected gap is reported with its own spread, because it is much smaller than the spread across
# splits and the Methods text must not read as if it were a decisive result.
s = lambda k: float(ab[k][0].split('± ')[1])
N['ablation_margin_within_spread'] = bool(abs(m(d0) - m(u0)) < min(s(d0), s(u0)))
_mn = lambda x: x.replace('-', '\u2212')               # typographic minus for the negative R2
# The three-split ablation puts directed marginally ahead, but the ten-split campaign (campaign/outputs/
# campaign_runs.csv, arms `base` and `base_undir` on identical splits) cannot resolve a difference in either
# direction, and the nominal winner changes with the selection rule. The sentence therefore reports the ablation and
# says the comparison is not resolved, rather than claiming an accuracy benefit the larger experiment does not show.
def _paired_dir_undir(path):
    """(mean, standard error, t, n) of undirected − directed test R², paired over (property, split)."""
    import csv as _csv, math as _math, statistics as _stats
    pairs = {'H2_N2': ('H2', 'H2/N2'), 'O2_N2': ('O2', 'O2/N2'), 'CO2_CH4': ('CO2', 'CO2/CH4')}
    best = {}
    for r in _csv.DictReader(open(path)):
        if r['arm'] in ('base', 'base_undir'):
            k = (r['arm'], r['pair'], int(r['seed'])); v = float(r['val_loss'])
            if k not in best or v < best[k][0]:
                best[k] = (v, r)
    diffs = []
    for pair, props in pairs.items():
        for p in props:
            for sd in range(1, 11):
                a, b = best.get(('base', pair, sd)), best.get(('base_undir', pair, sd))
                if a and b and a[1][f'{p}|r^2'] and b[1][f'{p}|r^2']:
                    diffs.append(float(b[1][f'{p}|r^2']) - float(a[1][f'{p}|r^2']))
    if len(diffs) < 10:
        return None
    mean = sum(diffs) / len(diffs); se = _stats.stdev(diffs) / _math.sqrt(len(diffs))
    return mean, se, mean / se, len(diffs)

_CAMP = os.path.join(os.path.dirname(O), 'campaign', 'outputs', 'campaign_runs.csv')
_pd = _paired_dir_undir(_CAMP) if os.path.exists(_CAMP) else None
N['ablation_dir_paired'] = None if _pd is None else {'mean': _pd[0], 'se': _pd[1], 't': _pd[2], 'n': _pd[3]}
N['ablation_text_directed'] = (
    f"In the ablation the directed model is nominally ahead of the undirected one (MAE {m(d0):.3f} against "
    f"{m(u0):.3f}, R² {q(d0):.2f} against {q(u0):.2f}), by a margin "
    f"{'smaller than the spread across splits' if N['ablation_margin_within_spread'] else 'larger than the spread across splits'} "
    f"(± {min(s(d0), s(u0)):.3f})" +
    ('' if _pd is None else
     f", and repeating the comparison on ten train/test splits does not resolve a difference in either direction "
     f"(paired over {_pd[3]} property–split pairs, t = {abs(_pd[2]):.1f})") +
    f". We therefore use directed message passing for the reason above rather than as an accuracy claim. Weighting "
    f"the aggregation by the transition probabilities, by contrast, is clearly harmful "
    f"(MAE {m(d1):.3f} against {m(d0):.3f}, R² {_mn(f'{q(d1):.2f}')} against {q(d0):.2f}).")
json.dump(N, open(os.path.join(O, 'numbers.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in N.items() if k not in ('table1_exact',)}, indent=1))
