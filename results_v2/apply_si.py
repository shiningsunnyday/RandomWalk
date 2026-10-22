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
               'groups were revised after this panel was drawn — eleven edited definitions, the two added '
               'groups, and a moved R site on P36 — and the definitions used throughout the work are the ones '
               'distributed with the code.')
log.append('Figure S1 caption: 99 groups + stale-panel note')

# ---- Table S10 (new): ablation of edge weights and message-passing direction, placed after Table S9
abl = json.load(open(os.path.join(O, 'ablation.json')))
s9_tbl = tables_after(t, 'Table S9')[-1]
cap10 = deepcopy(find_par(t, 'Table S9')); set_runs_text(cap10, abl['caption'])
for r in cap10.findall(W + 'r')[1:]: cap10.remove(r)
tbl10 = deepcopy(tables_after(t, 'Table S2')[0])
s9_tbl.addnext(cap10); cap10.addnext(tbl10)
spacer = deepcopy(find_par(t, 'Table S9')); [spacer.remove(r) for r in spacer.findall(W + 'r')]
s9_tbl.addnext(spacer)
hdr = tbl10.findall(W + 'tr')[0]
for tc, v in zip(hdr.findall(W + 'tc'), abl['header']): set_cell_text(tc, v)
fill_table_rows([tbl10], abl['rows'])
set_col_widths(tbl10, [2500, 1000, 1700, 1700])
for p in [cap10] + [p for tr in tbl10.findall(W + 'tr')[:-1] for p in tr.iter(W + 'p')]:   # keep caption + table on one page
    ppr = p.find(W + 'pPr')
    if ppr is None:
        ppr = etree.SubElement(p, W + 'pPr'); p.insert(0, ppr)
    if ppr.find(W + 'keepNext') is None:
        ppr.insert(1 if ppr.find(W + 'pStyle') is not None else 0, etree.Element(W + 'keepNext'))
log.append('Table S10 added: ' + json.dumps(abl['rows']))

# ---- Table S11 (new): where sampling fails, which Table 1 does not show (reviewer asked for failure cases)
fm = json.load(open(os.path.join(O, 'numbers.json')))['failure_modes']
cap11 = deepcopy(find_par(t, 'Table S9'))
set_runs_text(cap11, (
    'Table S11: Where ancestral sampling fails. Counts are for the seeded run of '
    f"{fm['n_sampled']} sampled trajectories that produced the 1000 structures behind the validity columns of "
    'Table 1. A repeat unit requires the random walk to return to the motif it started from, and a trajectory that '
    'instead runs out of probability mass, revisits its current motif or cannot be extended is discarded before any '
    f"chemistry is attempted; only {fm['closure_pct']}% of trajectories close. The validity columns of Table 1 are "
    'conditioned on that closure, so this table gives the unconditional picture. The last row is the share of all '
    'sampled trajectories that survive every stage.'))
for r in cap11.findall(W + 'r')[1:]: cap11.remove(r)
tbl11 = deepcopy(tables_after(t, 'Table S2')[0])
tbl10.addnext(cap11); cap11.addnext(tbl11)
spacer11 = deepcopy(find_par(t, 'Table S9')); [spacer11.remove(r) for r in spacer11.findall(W + 'r')]
tbl10.addnext(spacer11)
for tc, v in zip(tbl11.findall(W + 'tr')[0].findall(W + 'tc'), fm['header']): set_cell_text(tc, v)
fill_table_rows([tbl11], fm['rows'])
set_col_widths(tbl11, [1500, 2900, 1100, 1400])   # 'Count' wide enough for 115,068 on one line
for p in [cap11] + [p for tr in tbl11.findall(W + 'tr')[:-1] for p in tr.iter(W + 'p')]:
    ppr = p.find(W + 'pPr')
    if ppr is None:
        ppr = etree.SubElement(p, W + 'pPr'); p.insert(0, ppr)
    if ppr.find(W + 'keepNext') is None:
        ppr.insert(1 if ppr.find(W + 'pStyle') is not None else 0, etree.Element(W + 'keepNext'))
log.append('Table S11 added: ' + json.dumps(fm['rows']))

# ---- Tables S12/S13 (new): the prediction baselines the reviewers asked for, under the same protocol.
# S12 is each baseline as a practitioner would tune it (the log-permeability target is one of the hyperparameters
# chosen on validation); S13 restricts every baseline to the untransformed target our own model uses, so the two
# tables separate "better representation" from "better target transform".
SIT = os.path.join(H, 'baselines', 'outputs', 'si_tables.json')
if os.path.exists(SIT):
    bl = json.load(open(SIT))
    s9_frag = next(x for x in tables_after(t, 'Table S9') if len(x.findall(W + 'tr')[0].findall(W + 'tc')) == 7)
    anchor = tbl11
    CAPS = {
        'tuned': ('Table S12: Property-prediction baselines under the same protocol as Table 2. Every method is '
                  'fitted on the training split only, its hyperparameters are chosen by validation loss alone, and '
                  'the test split is only evaluated; metrics are the test MAE and R² on normalized values, '
                  'averaged over the same three train/test splits, written as MAE / R². Each baseline is also '
                  'offered the logarithm of permeability as a target, selected on validation like any other '
                  'hyperparameter, because that is how these methods would be used in practice; Table S13 repeats '
                  'the comparison with that option removed, which is the target our own model uses. An asterisk '
                  'marks a method for which at least one of the eighteen (property, split) fits has R² below '
                  '−1: a model fitted in log space can extrapolate above the largest training permeability on a '
                  'single test polymer, and exponentiating that error dominates a mean R² while barely moving '
                  'the MAE, so those rows should be read on MAE. A cell given as ≪0 is an R² below −10, that is, '
                  'a fit that diverged on that property; its MAE is reported as it stands. These numbers are not '
                  'comparable with the baseline comparison reported previously for this grammar (53): that '
                  'comparison predates the correction of several permeability labels in the dataset used here, and '
                  'its baselines were run at fixed hyperparameters rather than tuned on a validation split as they '
                  'are above. Both changes favour the baselines, and the ordering differs accordingly.'),
        'raw': ('Table S13: The same baselines as Table S12, restricted to the untransformed permeability and '
                'selectivity targets that our own model is trained on, so that the comparison isolates the effect of '
                'the representation from the effect of the target transform. Columns and selection rules are '
                'otherwise identical to Table S12.'),
    }
    for key in ('tuned', 'raw'):
        cap = deepcopy(find_par(t, 'Table S9')); set_runs_text(cap, CAPS[key])
        for r in cap.findall(W + 'r')[1:]: cap.remove(r)
        tbl = deepcopy(s9_frag)
        for tr in tbl.findall(W + 'tr')[2:]: tbl.remove(tr)          # keep header + one template row
        spacer = deepcopy(find_par(t, 'Table S9')); [spacer.remove(r) for r in spacer.findall(W + 'r')]
        anchor.addnext(cap); cap.addnext(tbl)
        anchor.addnext(spacer)
        for tc, v in zip(tbl.findall(W + 'tr')[0].findall(W + 'tc'), bl[key]['header']):
            set_cell_text(tc, v)
        fill_table_rows([tbl], bl[key]['rows'])
        set_col_widths(tbl, [2089, 1333, 1333, 1333, 1333, 1333, 1333])   # 'x.xx/y.yy' on one line
        anchor = tbl
        log.append(f"Table S{12 if key == 'tuned' else 13}: {len(bl[key]['rows'])} rows; "
                   f"not yet available: {bl[key]['missing']}")
else:
    log.append('Tables S12/S13 SKIPPED: baselines/outputs/si_tables.json not found')

# ---- Table S14 (new): generation baselines, with the 114 dataset monomers as a reference row. That row is the
# point of the table: Retro* tracks molecule size almost perfectly across these methods, so our own 54% cannot be
# read as "more synthesizable than known membrane polymers" when the known polymers themselves score 9%.
SGT = os.path.join(H, 'baselines', 'outputs', 'si_gen_table.json')
if os.path.exists(SGT):
    gt = json.load(open(SGT))
    sv = gt.get('size_vs_retro', {})
    cap = deepcopy(find_par(t, 'Table S9'))
    set_runs_text(cap, (
        'Table S14: Generation compared across methods. Each baseline draws 1000 molecules from its own '
        'generative distribution; our row is the first 1000 structures of the best-first enumeration, scored with '
        'the same metric definitions. The protocols differ because the methods do: a baseline samples '
        'independently from a learned distribution, whereas our generator enumerates the grammar\u2019s most '
        'probable walks in order, rejecting any closed walk whose cycle of motifs it has already produced. That is '
        'a deliberate property rather than a limitation of the comparison \u2014 the enumeration is systematic, '
        'covering the design space in decreasing order of the model\u2019s own confidence, and it is '
        'interpretable, since every structure comes with the walk that produced it and the probability the grammar '
        'assigned to it. The second row is not a generative model: it is the 114 experimentally '
        'characterised monomers of the dataset itself, scored the same way, and it is included because it '
        'calibrates the other rows. Novelty is measured against that dataset, so it is 0% there by construction. '
        'Retro* is a retrosynthesis success rate computed with the same planner and budget throughout, on a seeded '
        'subsample where a full pass was too expensive. Retro* is a bounded tree search over a fixed template '
        'library, so a failure is not evidence that a molecule cannot be made: the 114 dataset monomers have all '
        f"been synthesized and reported, yet the planner solves {sv.get('dataset_retro', '')} of them, at a median "
        f"of {sv.get('dataset_median_ha', '')} heavy atoms against our {sv.get('ours_retro', '')} at "
        f"{sv.get('ours_median_ha', '')}. Success accordingly correlates with the median heavy-atom count across "
        f"the rows with both quantities (r = {sv.get('r_text', '')}, n = {sv.get('n', '')}). The column is "
        'therefore informative in one direction, as positive evidence of accessibility where the search succeeds. '
        'Our structures sit at the small end by construction, because best-first enumeration expands the most '
        'probable walks first and those are short walks built from the motif pairs the grammar has seen most '
        'often, so they are assembled from few, frequently used blocks.'))
    for r in cap.findall(W + 'r')[1:]: cap.remove(r)
    tbl14 = deepcopy(s9_frag)
    for tr in tbl14.findall(W + 'tr')[2:]: tbl14.remove(tr)
    spacer = deepcopy(find_par(t, 'Table S9')); [spacer.remove(r) for r in spacer.findall(W + 'r')]
    anchor.addnext(cap); cap.addnext(tbl14)
    anchor.addnext(spacer)
    for tc, v in zip(tbl14.findall(W + 'tr')[0].findall(W + 'tc'), gt['header']):
        set_cell_text(tc, v)
    fill_table_rows([tbl14], gt['rows'])
    set_col_widths(tbl14, [3000, 1050, 1050, 1050, 1150, 1050, 1739])
    log.append(f"Table S14: {len(gt['rows'])} rows; not yet available: {gt['missing']}")
    anchor = tbl14
else:
    log.append('Table S14 SKIPPED: baselines/outputs/si_gen_table.json not found')

# ---- Table S15: the same comparison at ten splits. With about nineteen test polymers per split, a three-split
# mean is sensitive to which polymers land in the test set; re-running every method at ten splits is the cheapest
# check that the ordering is not an artefact of that choice.
K10 = os.path.join(H, 'baselines', 'outputs', 'k10_table.json')
if os.path.exists(K10) and 'tbl14' in dir():
    kt = json.load(open(K10))
    cap = deepcopy(find_par(t, 'Table S9'))
    set_runs_text(cap, (
        'Table S15: The comparison of Table S12 repeated with ten train/test splits instead of three, everything '
        'else identical. Metrics are averaged over the six properties on the normalized raw scale, so a method '
        'fitted in log space is scored after transforming its predictions back; median and worst are taken over '
        'the six per-property means. Ten splits leave every method less accurate in R\u00b2 than three did, which '
        'is what one expects when a three-split mean is drawn from about nineteen test polymers each time, and '
        f"the ordering by MAE is largely preserved. Ours ranks {kt['ours_rank']} of {kt['n']} by mean absolute "
        'error here. The split-to-split spread is the dominant uncertainty in this table and in Table 2: the '
        'worst single property-split fit of our own model has R\u00b2 below zero on the same split that is worst '
        'for several baselines, while its median across properties is comparable to the most accurate methods.'))
    for r in cap.findall(W + 'r')[1:]:
        cap.remove(r)
    tbl15 = deepcopy(s9_frag)
    for tr in tbl15.findall(W + 'tr')[2:]:
        tbl15.remove(tr)
    spacer = deepcopy(find_par(t, 'Table S9')); [spacer.remove(r) for r in spacer.findall(W + 'r')]
    anchor.addnext(cap); cap.addnext(tbl15)
    anchor.addnext(spacer)
    for tc, v in zip(tbl15.findall(W + 'tr')[0].findall(W + 'tc'), kt['header']):
        set_cell_text(tc, v)
    fill_table_rows([tbl15], [r + [''] * (7 - len(r)) for r in kt['rows']])
    set_col_widths(tbl15, [3400, 1150, 1150, 1150, 1150, 600, 600])
    anchor = tbl15
    log.append(f"Table S15 added: {len(kt['rows'])} rows, our rank {kt['ours_rank']}/{kt['n']}")
else:
    log.append('Table S15 SKIPPED: k10_table.json not found')

# The i.i.d.-sampling panels that stood here have been removed. Our generator is an enumeration, not a sampler:
# it returns the grammar's highest-confidence walks in order, which is what makes it systematic and interpretable,
# and scoring it as if it drew independent samples measures the wrong thing. Table S14 is the comparison we report
# — our best-first 1000 against each baseline's 1000 samples. The i.i.d. measurements are kept out of the paper but
# recorded in baselines/generation/unified/ and BRIEFING_2 section 10.

save(t, DOC)
open(os.path.join(O, 'apply_si.log'), 'w').write('\n'.join(log) + '\n')
print('\n'.join(log))
