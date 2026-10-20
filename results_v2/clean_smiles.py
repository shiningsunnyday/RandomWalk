"""
Clean, sanitized SMILES for generated walks, built exactly like the dataset SMILES (scripts/regen_smiles_diag.py:
motif-by-motif assembly along the chosen motif-graph edges, then every unconsumed R stand-in dropped and H-capped,
atom maps cleared). Validity = RDKit sanitization succeeds.

usage: python results_v2/clean_smiles.py <gen.json> <out.txt>
Output lines: '<clean_smiles> <0|1 sanitized> <walk>' in input order.
"""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__))
os.environ['GC_DIR'] = os.path.join(H, 'data', 'group-contrib')
os.environ['WALKS_FILE'] = os.path.join(H, 'data', 'group-contrib', 'polymer_walks_v2_preprocess.txt')
sys.path.insert(0, os.path.join(H, 'code', 'scripts'))
import networkx as nx
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import regen_smiles_diag as diag

recs = json.load(open(sys.argv[1]))
out = []
for r in recs:
    g = nx.DiGraph()
    for a, A, b, B, e in r['edges']:
        g.add_node(a, val=A); g.add_node(b, val=B)
        if b != 0:                                   # loop-back into the root is not an assembly step
            g.add_edge(a, b, e=e)
    diag.DAGS = {0: (g, None, None)}
    try:
        m, _ = diag.build(0, fix=True)
        smi, ok = diag.clean_smiles(diag.strip_unused_r(m))
        ok = ok and Chem.MolFromSmiles(smi) is not None
    except Exception as ex:
        smi, ok = 'BUILD_FAILED', False
    out.append(f"{smi} {int(ok)} {r['walk']}")
open(sys.argv[2], 'w').write('\n'.join(out) + '\n')
print(f"{sum(l.split()[1] == '1' for l in out)}/{len(out)} sanitized")
