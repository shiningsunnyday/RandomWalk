"""
Ensemble the per-member predictions written by predict.py: mean and np.std (population, ddof=0) over the three
test-seed members, exactly as summarize_test_walks.py (ben_preds.txt) and plot_group_contrib_with_err (out_*.txt) did.

usage: python repro_v1/ensemble.py <members.csv> [<out.txt>]
Output lines: '<walk> <property> <mean> +- <std>' in walk order, H2, H2/N2, O2, O2/N2, CO2, CO2/CH4 per walk.
"""
import sys, csv
import numpy as np
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
_seen = set()   # walks listed twice (dataset polymers sharing a motif sequence) predict identically: count once per model
rows = [r for r in rows if not ((r['predictor'], r['walk'], r['pair']) in _seen or _seen.add((r['predictor'], r['walk'], r['pair'])))]
vals = defaultdict(list)
order = []
for r in rows:
    gas, _ = r['pair'].split('/')
    for prop, v in ((gas, r['perm']), (r['pair'], r['sel'])):
        key = (r['walk'], prop)
        if key not in vals: order.append(key)
        vals[key].append(float(v))
walks = list(dict.fromkeys(w for w, _ in order))
props = ['H2', 'H2/N2', 'O2', 'O2/N2', 'CO2', 'CO2/CH4']
lines = []
for w in walks:
    for p in props:
        if (w, p) in vals:
            x = np.array(vals[(w, p)])
            assert len(x) == 3, (w, p, len(x))
            lines.append(f'{w} {p} {x.mean()} +- {x.std()}')
out = open(sys.argv[2], 'w') if len(sys.argv) > 2 else sys.stdout
out.write('\n'.join(lines) + '\n')
