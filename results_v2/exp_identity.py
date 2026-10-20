"""
Is any of the four synthesized polymers already in the dataset, and if so, was its label seen in training?

The text says PIM-PI-4 "has previously been synthesized and studied" (66). It is also row `PIM-PI-1` of the
114-structure dataset, which matters because that label was then available to the predictor. The dataset numbers its
PIM-PI series independently of ours, so the names prove nothing; this script checks the chemistry instead.

  1. build each experimental walk the way gen.py/clean_smiles.py build a generated one, and look for its canonical
     SMILES among the 114 dataset structures;
  2. our bracket notation does not say which of a motif's R sites a side chain uses, so for the one near-match it
     also enumerates the methyl attachment assignments and reports whether any reproduces the dataset structure;
  3. for each split seed, report whether that dataset row was in train, validation or test.

usage: cd results_v2/code && PYTHONPATH=<stubs> python ../exp_identity.py
"""
import os, sys, json, itertools
H = os.path.dirname(os.path.abspath(__file__))
GC = os.path.join(H, 'data', 'group-contrib')
os.environ['dataset'] = GC
os.environ['GC_DIR'] = GC
os.environ['WALKS_FILE'] = os.path.join(GC, 'polymer_walks_v2_preprocess.txt')
sys.path.insert(0, os.path.join(H, 'code'))
sys.path.insert(0, os.path.join(H, 'code', 'scripts'))
import networkx as nx
from utils.graph import verify_walk, load_mols, annotate_extra, r_member_lookup
import regen_smiles_diag as diag
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

G = nx.read_edgelist(os.path.join(GC, 'red_graph.edgelist'), create_using=nx.MultiDiGraph)
mols = load_mols(os.path.join(GC, 'all_groups') + '/')
annotate_extra(mols, os.path.join(GC, 'all_groups', 'all_extra.txt'))
r_lookup = r_member_lookup(mols)


def split_walk(w):
    """'->'-separated motifs at depth 0, bracketed side chains kept inside their token (as predict.py does)"""
    toks, buf, depth, i = [], '', 0, 0
    while i < len(w):
        if w.startswith('->', i) and depth == 0:
            toks.append(buf); buf = ''; i += 2; continue
        depth += (w[i] == '[') - (w[i] == ']'); buf += w[i]; i += 1
    return toks + [buf]


def spec_of(walk):
    _, conn = verify_walk(r_lookup, G, split_walk(walk), loop_back=True)
    return [(a.id, a.val.split(':')[0], b.id, b.val.split(':')[0], int(e)) for a, b, e in conn]


def build(spec):
    g = nx.DiGraph()
    for a, A, b, B, e in spec:
        g.add_node(a, val=A); g.add_node(b, val=B)
        if b != 0:                                   # loop-back into the root is not an assembly step
            g.add_edge(a, b, e=e)
    diag.DAGS = {0: (g, None, None)}
    m, _ = diag.build(0, fix=True)
    smi, ok = diag.clean_smiles(diag.strip_unused_r(m))
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if ok else None


ds, inv = {}, {}
for line in open(os.path.join(GC, 'polymer_walks_v2_smiles.txt')):
    if line.strip():
        t = line.split(); ds[t[-1]] = Chem.MolToSmiles(Chem.MolFromSmiles(t[0]))
for name, smi in ds.items():
    inv.setdefault(smi, []).append(name)

exp = json.load(open(os.path.join(H, 'inputs', 'experimental_walks.json')))
for name, walk in exp.items():
    spec = spec_of(walk)
    can = build(spec)
    f = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(can))
    print(f'{name:10s} {f:14s} dataset match: {inv.get(can, "none")}')
    if inv.get(can):
        continue
    # the notation leaves the side-chain attachment points open: try the alternatives before calling it novel
    side = [i for i, s in enumerate(spec) if s[3] == 'P17']
    if not side:
        continue
    opts = sorted(int(k) for k in G[spec[side[0]][1]][spec[side[0]][3]])
    reached = {}
    for combo in itertools.permutations(opts, len(side)):
        sp = list(spec)
        for j, e in zip(side, combo):
            sp[j] = sp[j][:4] + (e,)
        try:
            s = build(sp)
        except Exception:
            continue
        if s and s not in reached:
            reached[s] = combo
    hits = {inv[s][0]: c for s, c in reached.items() if s in inv}
    print(f'{"":10s} {len(reached)} distinct structures over the {len(side)} side-chain attachment points; '
          f'dataset match under an alternative assignment: {hits if hits else "none"}')

print()
sys.path.insert(0, os.path.join(H, 'baselines'))
import bl_data as D
for row in sorted({n for s, ns in inv.items() for n in ns} & {n for n in ds} & set(
        n for n in ds if n in [x for s in inv for x in inv[s]]) & {'PIM-PI-1'}):
    for pair in D.pairs():
        names = [r['name'] for r in D.masked(pair)]
        if row not in names:
            print(f'{pair:4s} {row} is not in the masked set for this pair'); continue
        i = names.index(row)
        for seed in D.SEEDS:
            tr, va, te = D.split(pair, seed)
            where = 'train' if i in tr else ('validation' if i in va else 'test' if i in te else '??')
            print(f'{pair:4s} seed {seed}: dataset row {row} is in {where}')
