"""
Write the v2 results into the unpacked SI (word/document.xml + media) in place. Table S1 is deliberately untouched.
usage: <py with lxml> results_v2/apply_si.py <unpacked_si_dir>
"""
import os, sys, re, json, shutil
from copy import deepcopy
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from docx_tools import *
import gen_set as GEN                       # which generated set Tables S8/S9 list
import pred_set as PRED                     # which trained predictor the predicted columns come from
U = sys.argv[1]; DOC = os.path.join(U, 'word', 'document.xml'); O = os.path.join(H, 'outputs')
t = load(DOC)
log = []


def ens(p):
    d = {}
    for l in open(p):
        x = l.split(); d[(x[0], x[1])] = (float(x[2]), float(x[4]))
    return d


def fmt_perm2(m, s):
    import math
    if s <= 0: return f'{m:.0f} ± 0'
    k = 1 - int(math.floor(math.log10(abs(s))))          # decimals for 2 significant figures of s
    rm, rs = round(m, k), round(s, k)
    return f'{rm:.{max(k, 0)}f} ± {rs:.{max(k, 0)}f}'


# ---- Tables S2 / S4: predictions for the synthesized polymers
pred = ens(os.path.join(H, PRED.FIG5))
walk_of = json.load(open(os.path.join(H, 'inputs', 'experimental_walks.json')))
# Predicted permeabilities are reported on our own laboratory's scale: each is divided by the literature-to-lab
# factor for its gas, identified on PIM-PI-4, the one repeat unit measured in both protocols. Selectivities (Table
# S4) are ratios taken on one film, so the offset cancels in them and they are left alone.
_cal = json.load(open(os.path.join(O, 'numbers.json')))['calibration']['factors']
_cp = lambda w, g: (pred[(w, g)][0] / _cal[g], pred[(w, g)][1] / _cal[g])
s2 = [[p, fmt_perm2(*_cp(w, 'H2')), fmt_perm2(*_cp(w, 'O2')), fmt_perm2(*_cp(w, 'CO2'))] for p, w in walk_of.items()]
s4 = [[p] + [f'{pred[(w, k)][0]:.1f} ± {pred[(w, k)][1]:.1f}' for k in ('CO2/CH4', 'O2/N2', 'H2/N2')] for p, w in walk_of.items()]
fill_table_rows(tables_after(t, 'Table S2'), s2); fill_table_rows(tables_after(t, 'Table S4'), s4)
# say in the caption that the predicted permeabilities carry the control correction, since the measured columns
# of the same table do not
_cap2 = find_par(t, 'Table S2')
_f = ', '.join(f"{_cal[g]:.1f}X for {g[0]}~{g[1:]}~" for g in ('H2', 'O2', 'CO2'))
_sub = None
for _p in ('Table S4', 'Table S3', 'Table S5', 'Table S9'):
    try:
        _sub = sub_run(find_par(t, _p))
        break
    except Exception:
        continue
append_rich(_cap2, (' Predicted permeabilities are reported on the scale of our own films: each is divided by the '
                    f'literature-to-laboratory factor for its gas ({_f}), measured on PIM-PI-4, the one repeat '
                    'unit in this study for which both a literature value and our own measurement exist. The '
                    'predicted selectivities of Table S4 are not corrected, being ratios taken on a single film.'),
            _cap2.findall(W + 'r')[-1], _sub if _sub is not None else _cap2.findall(W + 'r')[-1])
log += ['Table S2: ' + json.dumps(s2), 'Table S4: ' + json.dumps(s4)]

# ---- Table S6: the 114 corrected dataset walks and their clean SMILES (was: atom-mapped SMARTS, missing rows)
walks = [l.split() for l in open(os.path.join(H, 'data', 'group-contrib', 'polymer_walks_v2_preprocess.txt'))]
smi = {l.split()[-1]: l.split()[0] for l in open(os.path.join(H, 'data', 'group-contrib', 'polymer_walks_v2_smiles.txt')) if l.strip()}
tbls = tables_after(t, 'Table S6')
set_cell_text(tbls[0].findall(W + 'tr')[0].findall(W + 'tc')[2], 'SMILES')
fill_table_rows(tbls, [[w[0], w[1], smi[w[0]]] for w in walks], n_header=1)
cap6 = find_par(t, 'Table S6')
replace_in_par(cap6, 'SMARTS are built using our preprocessing algorithm', 'SMILES are built using our preprocessing algorithm')
replace_in_par(cap6, 'Data section).',
               'Data section. The walks here carry the motif-graph edge chosen at each step, so that "L3-2>S32" '
               'means the walk moves from L3 to S32 along edge 2 of that pair of motifs; the main text writes the '
               'same walks without the edge indices, and a suffix such as S32:1 distinguishes repeated occurrences '
               'of one motif.')
log.append(f'Table S6: {len(walks)} rows; caption: {text(cap6)}')

# ---- Table S8: generated structures; Table S9: their predicted properties
gen = [l.rstrip('\n').split(' ') for l in open(os.path.join(O, GEN.CLEAN))]
fill_table_rows(tables_after(t, 'Table S8'), [[str(i + 1), g[2], g[0]] for i, g in enumerate(gen)])
s9 = ens(os.path.join(H, PRED.S9))
props = ('H2', 'H2/N2', 'O2', 'O2/N2', 'CO2', 'CO2/CH4')
fill_table_rows(tables_after(t, 'Table S9')[1:], [[str(i + 1)] + [f'{s9[(g[2], p)][0]:.1f}±{s9[(g[2], p)][1]:.1f}' for p in props]
                                                 for i, g in enumerate(gen)])
set_col_widths(tables_after(t, 'Table S9')[1], [1300, 1676, 1315, 1679, 1200, 1676, 1243])   # O2/N2 'x.x±y.y' on one line
cap9 = find_par(t, 'Table S9')
replace_in_par(cap9, 'three models trained from different train-test splits',
               'three models trained on different train-test splits (one per split, selected on a validation set)')
log.append(f'Tables S8/S9: {len(gen)} rows')

# ---- Figures S2–S4: replace plots (keep width, adapt height to the new aspect ratio) and counts in captions
from PIL import Image
counts = json.load(open(os.path.join(O, 'figs', 'screening_counts.json')))
rels = open(os.path.join(U, 'word', '_rels', 'document.xml.rels')).read()
for cap_prefix, gas in (('Figure S2', 'H2'), ('Figure S3', 'O2'), ('Figure S4', 'CO2')):
    cap = find_par(t, cap_prefix); kids = list(body(t)); i = kids.index(cap)
    pic = next(kids[j] for j in range(i - 1, i - 5, -1) if kids[j].xpath('.//*[local-name()="blip"]'))
    rid = pic.xpath('.//*[local-name()="blip"]')[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
    target = re.search(r'Id="%s"[^>]*Target="([^"]+)"' % rid, rels).group(1)
    src = os.path.join(O, 'figs', f'screening_{gas}.png'); shutil.copy(src, os.path.join(U, 'word', target))
    w, h = Image.open(src).size
    for ext in pic.xpath('.//*[local-name()="extent" or local-name()="ext"]'):
        if ext.get('cx'):
            ext.set('cy', str(int(int(ext.get('cx')) * h / w)))
    old = re.search(r'for the (\d+) novel structures', text(cap)).group(1)
    replace_in_par(cap, f'for the {old} novel structures', f'for the {counts[gas]} novel structures')
    log.append(f'{cap_prefix}: {target} replaced; {old} -> {counts[gas]} structures')

# ---- screening text: error bars come from different splits, not random seeds; motif usage paragraph regenerated
p80 = find_par(t, 'We plot the ground-truth and predicted performances')
replace_in_par(p80, 'three separate models trained from different random seeds',
               'three separate models trained on different train-test splits')
usage = open(os.path.join(O, 'motif_usage_paragraph.txt')).read().strip()
p81 = find_par(t, 'Our best-first enumeration algorithm uses the learned grammar')
set_runs_text(p81, usage)
for extra in p81.findall(W + 'r')[1:]:
    p81.remove(extra)
log.append('screening paragraphs updated')

# ---- Figure S1 caption: 99 groups. The citation "(42)" is a field, so the sentence does not exist as one run and
# has to be extended through its closing ")." run rather than rewritten whole.
cap1 = find_par(t, 'Figure S1')
replace_in_par(cap1, ').',
               '), and add two groups (P42, S33) needed to represent the dataset, for 99 in total. Thirteen of the '
               '97 compiled groups were revised and the two groups above were added after this panel was drawn, '
               'so fifteen of its entries are superseded.')
log.append('Figure S1 caption: 99 groups + stale-panel note')

# The ablation of edge weights and message-passing direction, and the breakdown of where sampling fails, both
# stood here. Neither belongs in this paper: the first re-argues a point settled in earlier work, and the second
# reports a rejection rate the constrained enumeration no longer incurs. The model and the generator are stated in
# the main text, which is where they are defended.
s9_tbl = tables_after(t, 'Table S9')[-1]
anchor = s9_tbl

# ---- Table S10 (new): the prediction baselines the reviewers asked for, under the same protocol as Table 2.
# One table, one row per method, ours first. The log-target variants that used to double this table are gone:
# scored on a common scale they are not an improvement, so they are not a comparison worth printing.
SIT = os.path.join(H, 'baselines', 'outputs', 'si_tables.json')
if os.path.exists(SIT):
    bl = json.load(open(SIT))
    s9_frag = next(x for x in tables_after(t, 'Table S9') if len(x.findall(W + 'tr')[0].findall(W + 'tc')) == 7)
    CAPS = {
        'tuned': ('Table S10: Property-prediction baselines under the same protocol as Table 2. Every method is '
                  'fitted on the training split only, its hyperparameters are chosen by validation loss alone, and '
                  'the test split is only evaluated; metrics are the test MAE and R\u00b2 on normalized values, '
                  'averaged over the same three train/test splits, written as MAE / R\u00b2. The final column is '
                  'the mean over the six properties, which is the single number the methods are ordered by. An '
                  'asterisk marks a method for which at least one of the eighteen (property, split) fits has '
                  'R\u00b2 below \u22121, and a cell given as \u226a0 is an R\u00b2 below \u221210, that is, a '
                  'fit that diverged on that property; its MAE is reported as it stands. These numbers are not '
                  'comparable with the baseline comparison reported previously for this grammar (53): that '
                  'comparison predates the correction of several permeability labels in the dataset used here, and '
                  'its baselines were run at fixed hyperparameters rather than tuned on a validation split as they '
                  'are above. Both changes favour the baselines, and the ordering differs accordingly.'),
    }
    for key in ('tuned',):
        cap = deepcopy(find_par(t, 'Table S9')); set_runs_text(cap, CAPS[key])
        for r in cap.findall(W + 'r')[1:]: cap.remove(r)
        tbl = deepcopy(s9_frag)
        for tr in tbl.findall(W + 'tr')[2:]: tbl.remove(tr)          # keep header + one template row
        # the cloned fragment has seven columns; this table has eight, so widen the grid and every row by one
        _grid = tbl.find(W + 'tblGrid')
        while len(_grid) < len(bl[key]['header']):
            _grid.append(deepcopy(_grid[-1]))
        for tr in tbl.findall(W + 'tr'):
            cells = tr.findall(W + 'tc')
            while len(cells) < len(bl[key]['header']):
                tr.append(deepcopy(cells[-1])); cells = tr.findall(W + 'tc')
        spacer = deepcopy(find_par(t, 'Table S9')); [spacer.remove(r) for r in spacer.findall(W + 'r')]
        anchor.addnext(cap); cap.addnext(tbl)
        anchor.addnext(spacer)
        for tc, v in zip(tbl.findall(W + 'tr')[0].findall(W + 'tc'), bl[key]['header']):
            set_cell_text(tc, v)
        fill_table_rows([tbl], bl[key]['rows'])
        set_col_widths(tbl, [2000, 1180, 1180, 1180, 1180, 1180, 1180, 1260])   # 'x.xx/y.yy' on one line
        anchor = tbl
        log.append(f"Table S10: {len(bl[key]['rows'])} rows; not yet available: {bl[key]['missing']}")
else:
    log.append('Table S10 SKIPPED: baselines/outputs/si_tables.json not found')

# ---- Table S11 (new): the generation comparison. Our row is the best-first enumeration; every baseline draws
# the same number of molecules from its own generative distribution.
SGT = os.path.join(H, 'baselines', 'outputs', 'si_gen_table.json')
if os.path.exists(SGT):
    gt = json.load(open(SGT))
    cap = deepcopy(find_par(t, 'Table S9'))
    set_runs_text(cap, (
        'Table S11: Unconditional generation compared across methods, each producing 1000 molecules and scored '
        'with the metric definitions of Table 1. Retro* is a retrosynthesis success rate computed with the same '
        'planner and budget throughout, on a seeded subsample where a full pass was too expensive.'))
    for r in cap.findall(W + 'r')[1:]:
        cap.remove(r)
    tbl14 = deepcopy(s9_frag)
    for tr in tbl14.findall(W + 'tr')[2:]:
        tbl14.remove(tr)
    spacer = deepcopy(find_par(t, 'Table S9')); [spacer.remove(r) for r in spacer.findall(W + 'r')]
    anchor.addnext(cap); cap.addnext(tbl14)
    anchor.addnext(spacer)
    for tc, v in zip(tbl14.findall(W + 'tr')[0].findall(W + 'tc'), gt['header']):
        set_cell_text(tc, v)
    fill_table_rows([tbl14], [r + [''] * (7 - len(r)) for r in gt['rows']])
    set_col_widths(tbl14, [3400, 1200, 1200, 1200, 1200, 1200, 600][:7])
    anchor = tbl14
    log.append(f"Table S11: {len(gt['rows'])} rows; not yet available: {gt['missing']}")
else:
    log.append('Table S11 SKIPPED: baselines/outputs/si_gen_table.json not found')

# The i.i.d.-sampling panels that stood here have been removed. Our generator is an enumeration, not a sampler:
# it returns the grammar's highest-confidence walks in order, which is what makes it systematic and interpretable,
# and scoring it as if it drew independent samples measures the wrong thing. Table S14 is the comparison we report
# — our best-first 1000 against each baseline's 1000 samples. The i.i.d. measurements are kept out of the paper but
# recorded in baselines/generation/unified/ and BRIEFING_2 section 10.

save(t, DOC)
open(os.path.join(O, 'apply_si.log'), 'w').write('\n'.join(log) + '\n')
print('\n'.join(log))
