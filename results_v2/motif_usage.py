"""
Motif usage among the high-confidence generated structures (same ensemble-std thresholds as Figs S2–S4), for the SI
screening paragraph. Occurrences count every motif in a walk except the repeated root that closes the loop.
usage: python results_v2/motif_usage.py   -> outputs/motif_usage.json, outputs/motif_usage_paragraph.txt
"""
import os, re, json
from collections import Counter
H = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(H, 'outputs')
TH = {'H2': ('H2/N2', 500, 10), 'O2': ('O2/N2', 100, 2), 'CO2': ('CO2/CH4', 1000, 5)}
mot = lambda w: re.findall(r'[PSL]\d+', w)[:-1]
s9 = {}
for l in open(os.path.join(O, 's9_ensemble.txt')):
    x = l.split(); s9[(x[0], x[1])] = (float(x[2]), float(x[4]))
walks = [l.split()[0] for l in open(os.path.join(O, 'screening_walks.txt'))]
data_walks = [l.split()[1] for l in open(os.path.join(H, 'data/group-contrib/polymer_walks_v2_preprocess.txt'))]
in_data = Counter(m for w in data_walks for m in set(mot(w)))
res = {}
for gas, (pair, tp, ts) in TH.items():
    hc = [w for w in walks if s9[(w, gas)][1] < tp and s9[(w, pair)][1] < ts and s9[(w, gas)][0] > 0 and s9[(w, pair)][0] > 0]
    occ = Counter(m for w in hc for m in mot(w))
    struct = Counter(m for w in hc for m in set(mot(w)))
    res[gas] = dict(pair=pair, n=len(hc), unique=len(occ),
                    top=[(m, occ[m], struct[m], in_data.get(m, 0)) for m, _ in occ.most_common(5)])
json.dump(res, open(os.path.join(O, 'motif_usage.json'), 'w'), indent=1)
sub = lambda s: s.replace('H2', 'H₂').replace('N2', 'N₂').replace('O2', 'O₂').replace('CO2', 'CO₂').replace('CH4', 'CH₄')
parts = []
for gas in ('H2', 'O2', 'CO2'):
    r = res[gas]
    tops = ', '.join(f'{m} ({o} times, in {s} structures)' for m, o, s, _ in r['top'])
    parts.append(f"For {sub(r['pair'])}, the most commonly used motifs were {tops}.")
rare = sorted({m for g in res.values() for m, _, _, d in g['top'] if d <= 2})
rare_txt = (' Among these, ' + ', '.join(f'{m} (in {in_data.get(m, 0)} dataset polymer{"s" if in_data.get(m, 0) != 1 else ""})' for m in rare)
            + ' are rare in the training data, so the grammar does not only reuse the most frequent motifs.') if rare else ''
para = ('Our best-first enumeration algorithm uses the learned grammar to guide the exploration of the design space. '
        'However, this can result in a large degree of extrapolation of our predictive model. Thus, we use conservative '
        'error bars to filter the novel structures. '
        f"Out of the {res['H2']['n']}, {res['O2']['n']}, and {res['CO2']['n']} high-confidence novel structures for "
        f"H₂/N₂, O₂/N₂, and CO₂/CH₄ separations, respectively, a total of {res['H2']['unique']}, {res['O2']['unique']}, "
        f"and {res['CO2']['unique']} unique motifs were used. " + ' '.join(parts) + rare_txt)
open(os.path.join(O, 'motif_usage_paragraph.txt'), 'w').write(para + '\n')
print(para)
