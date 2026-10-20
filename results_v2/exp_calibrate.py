"""
Calibrate the predictions on the one validated polymer that is in the dataset -> outputs/calibration.json

Our PIM-PI-4 is dataset row `PIM-PI-1` (verified structurally in `exp_identity.py`), so for that one repeat unit we
have both a literature permeability, which is the kind of label the predictor was trained on, and our own
measurement. Their ratio is a direct, model-free estimate of the offset between the literature protocol and ours.
This script uses that offset to calibrate the predictions for the *other three* polymers, which are genuinely novel
and appear nowhere in the training data, and asks whether the systematic overprediction survives the correction.

  calibration factor   c_p  = literature_p(reference) / measured_p(reference)          one number per property
  calibrated prediction      = predicted_p(polymer) / c_p

The test is held out in the sense that matters: c_p is fitted on the reference polymer only and evaluated on three
polymers that contributed nothing to it. A single reference gives no error bar on c_p itself, which is the main
limitation and is reported alongside the result.

usage: python results_v2/exp_calibrate.py            -> outputs/calibration.{json,md}
"""
import os, sys, json
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(H, 'data', 'group-contrib')
O = os.path.join(H, 'outputs')
sys.path.insert(0, H)
from plots import EXP, load_ens                                                      # noqa: E402

PERM = ['H2', 'O2', 'CO2']
SEL = ['H2/N2', 'O2/N2', 'CO2/CH4']
PROPS = PERM + SEL
REF = 'PIM-PI-4'                       # our name for the polymer that is dataset row PIM-PI-1
LIT_ROW = 'PIM-PI-1'
NOVEL = ['PIM-PI-1', 'PIM-PI-2', 'PIM-PI-3']


def literature():
    """the dataset's own label for the reference repeat unit, as permeabilities and ideal selectivities"""
    line = next(l for l in open(os.path.join(D, 'polymer_walks_v2_preprocess.txt')) if l.split()[0] == LIT_ROW)
    vals = [float(x) for x in line.split()[-1].strip('(').rstrip(')').split(',')]
    c = dict(zip(('H2', 'N2', 'O2', 'CH4', 'CO2'), vals))
    lit = {g: c[g] for g in PERM}
    lit['H2/N2'] = c['H2'] / c['N2']
    lit['O2/N2'] = c['O2'] / c['N2']
    lit['CO2/CH4'] = c['CO2'] / c['CH4']
    return lit


def main():
    walk_of = json.load(open(os.path.join(H, 'inputs', 'experimental_walks.json')))
    pred = load_ens(os.path.join(O, 'fig5_ensemble.txt'))
    lit = literature()
    cal = {p: lit[p] / EXP[REF][p][0] for p in PROPS}

    out = {'reference': {'ours': REF, 'dataset_row': LIT_ROW,
                         'literature': lit, 'measured': {p: EXP[REF][p][0] for p in PROPS},
                         'calibration_factor': cal},
           'polymers': {}, 'summary': {}}

    for name, walk in walk_of.items():
        row = {}
        for p in PROPS:
            mu, sd = pred[(walk, p)]
            obs, err = EXP[name][p]
            row[p] = {'predicted': mu, 'pred_sd': sd, 'measured': obs, 'meas_err': err,
                      'ratio_raw': mu / obs, 'ratio_cal': (mu / cal[p]) / obs,
                      'calibrated': mu / cal[p],
                      # "agrees" = the calibrated prediction is inside measurement error widened by the ensemble sd,
                      # which is the same tolerance Figure 5 draws
                      'within_raw': abs(mu - obs) <= err + sd,
                      'within_cal': abs(mu / cal[p] - obs) <= err + sd / cal[p]}
        out['polymers'][name] = row

    # the held-out question: over the three novel polymers only, is the systematic bias removed?
    for tag, group in (('novel_three', NOVEL), ('reference_only', [REF]), ('all_four', list(walk_of))):
        s = {}
        for kind, props in (('permeability', PERM), ('selectivity', SEL)):
            raw = [out['polymers'][n][p]['ratio_raw'] for n in group for p in props]
            cl = [out['polymers'][n][p]['ratio_cal'] for n in group for p in props]
            s[kind] = {
                'n': len(raw),
                'geomean_ratio_raw': float(np.exp(np.mean(np.log(raw)))),
                'geomean_ratio_cal': float(np.exp(np.mean(np.log(cl)))),
                'mean_abs_log10_raw': float(np.mean(np.abs(np.log10(raw)))),
                'mean_abs_log10_cal': float(np.mean(np.abs(np.log10(cl)))),
                'n_within_raw': int(sum(out['polymers'][n][p]['within_raw'] for n in group for p in props)),
                'n_within_cal': int(sum(out['polymers'][n][p]['within_cal'] for n in group for p in props)),
                'range_raw': [float(min(raw)), float(max(raw))],
                'range_cal': [float(min(cl)), float(max(cl))]}
        out['summary'][tag] = s

    json.dump(out, open(os.path.join(O, 'calibration.json'), 'w'), indent=1)

    L = ['# Calibrating on the one validated polymer that is in the dataset', '',
         f'Reference: our **{REF}** = dataset row **{LIT_ROW}**. Calibration factor is literature / measured on that',
         'polymer alone; the three novel polymers contributed nothing to it.', '',
         '| property | literature | our measurement | calibration factor |', '|---|---|---|---|']
    for p in PROPS:
        L.append(f'| {p} | {lit[p]:.3g} | {EXP[REF][p][0]:.3g} | **{cal[p]:.2f}×** |')
    L += ['', '## The three novel polymers: predicted / measured, before and after', '',
          '| polymer | ' + ' | '.join(PROPS) + ' |', '|---' * (len(PROPS) + 1) + '|']
    for n in NOVEL + [REF]:
        r = out['polymers'][n]
        tag = f'**{n}**' if n != REF else f'{n} *(reference)*'
        L.append(f'| {tag} raw | ' + ' | '.join(f"{r[p]['ratio_raw']:.2f}" for p in PROPS) + ' |')
        L.append(f'| {tag} calibrated | ' + ' | '.join(f"{r[p]['ratio_cal']:.2f}" for p in PROPS) + ' |')
    L += ['', '## Per property, over the three novel polymers', '',
          '| property | factor c | geomean ratio raw | geomean ratio calibrated | calibrated range | within error |',
          '|---|---|---|---|---|---|']
    for p in PROPS:
        raw = [out['polymers'][n][p]['ratio_raw'] for n in NOVEL]
        cl = [out['polymers'][n][p]['ratio_cal'] for n in NOVEL]
        wr = sum(out['polymers'][n][p]['within_raw'] for n in NOVEL)
        wc = sum(out['polymers'][n][p]['within_cal'] for n in NOVEL)
        L.append(f'| {p} | {cal[p]:.2f}× | {np.exp(np.mean(np.log(raw))):.2f} | '
                 f'**{np.exp(np.mean(np.log(cl))):.2f}** | {min(cl):.2f}–{max(cl):.2f} | {wr}/3 → **{wc}/3** |')
    out['per_property'] = {p: {'c': cal[p],
                               'geomean_raw': float(np.exp(np.mean(np.log([out['polymers'][n][p]['ratio_raw']
                                                                           for n in NOVEL])))),
                               'geomean_cal': float(np.exp(np.mean(np.log([out['polymers'][n][p]['ratio_cal']
                                                                           for n in NOVEL])))),
                               'within_raw': int(sum(out['polymers'][n][p]['within_raw'] for n in NOVEL)),
                               'within_cal': int(sum(out['polymers'][n][p]['within_cal'] for n in NOVEL))}
                           for p in PROPS}
    json.dump(out, open(os.path.join(O, 'calibration.json'), 'w'), indent=1)

    s = out['summary']['novel_three']
    L += ['', '## Is the systematic bias removed? (three novel polymers only)', '',
          '| | geometric-mean ratio | mean &#124;log₁₀ ratio&#124; | within error | range |', '|---|---|---|---|---|']
    for kind in ('permeability', 'selectivity'):
        k = s[kind]
        L.append(f"| {kind}, raw | {k['geomean_ratio_raw']:.2f} | {k['mean_abs_log10_raw']:.3f} | "
                 f"{k['n_within_raw']}/{k['n']} | {k['range_raw'][0]:.2f}–{k['range_raw'][1]:.2f} |")
        L.append(f"| {kind}, **calibrated** | **{k['geomean_ratio_cal']:.2f}** | **{k['mean_abs_log10_cal']:.3f}** | "
                 f"**{k['n_within_cal']}/{k['n']}** | {k['range_cal'][0]:.2f}–{k['range_cal'][1]:.2f} |")
    open(os.path.join(O, 'calibration.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
