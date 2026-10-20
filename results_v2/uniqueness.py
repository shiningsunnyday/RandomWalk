"""
What the Unique column of Table 1 counts, and where the other 38% goes -> outputs/gen/uniqueness.json.

A repeat unit is a *closed* walk: it returns to the motif it started from. The best-first enumeration therefore emits
one entry per starting motif, so the same cyclic repeat unit appears up to k times for k motifs. This script peels the
redundancy apart, from the loosest to the strictest notion of "the same structure":

  walks            distinct walk strings (what the enumeration ranks)
  motif graphs     distinct motif graphs under a real DAG isomorphism (node labels and attachment points)
  cycles           distinct closed walks up to rotation and reflection -- the natural canonical form of a repeat unit
  label multisets  distinct multisets of motif labels: the comparison that utils.graph.is_novel degenerates into,
                   because dag_to_graph adds nodes but never edges
  raw SMILES       distinct atom-mapped SMILES straight from the assembler (what v1 reported as Unique)
  molecules        distinct canonical SMILES after cleaning (unconsumed R stand-ins dropped and H-capped) -- Table 1

usage: python results_v2/uniqueness.py
"""
import os, json, collections
import networkx as nx
from networkx.algorithms.isomorphism import is_isomorphic
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
H = os.path.dirname(os.path.abspath(__file__)); G = os.path.join(H, 'outputs', 'gen')


def cycle(walk):
    """the closed walk as a cycle of motifs, canonical over rotation and reflection"""
    s = walk.split('->')
    assert s[0] == s[-1], walk
    s = s[:-1]
    rots = [tuple(s[i:] + s[:i]) for i in range(len(s))]
    return min(rots + [tuple(reversed(r)) for r in rots])


def motif_graph(r):
    g = nx.DiGraph()
    for a, A, b, B, e in r['edges']:
        g.add_node(a, val=A); g.add_node(b, val=B); g.add_edge(a, b, e=e)
    return g


def n_graphs(recs):
    """distinct motif graphs; the Weisfeiler-Lehman hash is only a prefilter for the isomorphism test"""
    nm = lambda x, y: x['val'] == y['val']; em = lambda x, y: x['e'] == y['e']
    buckets = collections.defaultdict(list)
    for r in recs:
        g = motif_graph(r)
        b = buckets[nx.weisfeiler_lehman_graph_hash(g, node_attr='val', edge_attr='e')]
        if not any(is_isomorphic(g, g2, node_match=nm, edge_match=em) for g2 in b):
            b.append(g)
    return sum(len(b) for b in buckets.values())


def labels(r):
    v = {}
    for a, A, b, B, e in r['edges']:
        v[a] = A; v[b] = B
    return tuple(sorted(v.values()))


recs = json.load(open(os.path.join(G, 'bestfirst_n1400.json')))
clean = [l.rstrip('\n').split(' ')[0] for l in open(os.path.join(G, 'bestfirst_n1400_clean.txt'))]
can = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in clean]
assert [c for c in can[:1000]] == [Chem.MolToSmiles(Chem.MolFromSmiles(l.split(' ')[0]))
                                   for l in open(os.path.join(G, 'bestfirst_n1000_clean.txt'))], \
    'the 1400 enumeration does not reproduce the 1000 it extends'

out = {'levels': {}, 'marginal': {}}
for n in (1000, 1400):
    R, C = recs[:n], can[:n]
    out['levels'][n] = {'walks': len({r['walk'] for r in R}), 'motif_graphs': n_graphs(R),
                        'cycles': len({cycle(r['walk']) for r in R}), 'label_multisets': len({labels(r) for r in R}),
                        'raw_smiles': len({r['raw_smiles'] for r in R}), 'molecules': len(set(C)),
                        'n': n, 'unique_pct': 100 * len(set(C)) / n,
                        'motifs_per_walk': dict(collections.Counter(len(r['walk'].split('->')) - 1 for r in R))}
# the marginal yield of enumerating deeper: does best-first order keep finding new structures?
first, extra = set(can[:1000]), set(can[1000:1400]) - set(can[:1000])
out['marginal'] = {'walks': 400, 'new_molecules': len(extra), 'rate_pct': 100 * len(extra) / 400,
                   'needed_for_1000': None if len(extra) == 0 else
                   int(1400 + (1000 - len(first | extra)) / (len(extra) / 400))}
# how many distinct molecules come from more than one walk, and how many of those groups are pure rotations
grp = collections.defaultdict(list)
for i, c in enumerate(can[:1000]):
    grp[c].append(i)
dups = {k: v for k, v in grp.items() if len(v) > 1}
out['duplicate_groups'] = {
    'n_molecules_from_several_walks': len(dups), 'n_walks_in_such_a_group': sum(len(v) for v in dups.values()),
    'rotation_only': sum(len({cycle(recs[i]['walk']) for i in v}) == 1 for v in dups.values()),
    'sizes': dict(collections.Counter(len(v) for v in dups.values()))}
# the unique-cycle enumeration (gen_unique.py): the same best-first search, but a closed walk whose canonical cycle
# has already been accepted is rejected, so no two accepted walks are rotations or reflections of one another
UQ = os.path.join(G, 'unique_n1000.json')
if os.path.exists(UQ):
    uR = json.load(open(UQ))
    uC = [Chem.MolToSmiles(Chem.MolFromSmiles(l.rstrip('\n').split(' ')[0]))
          for l in open(os.path.join(G, 'unique_n1000_clean.txt'))]
    assert len({cycle(r['walk']) for r in uR}) == len(uR), 'the unique run has duplicate cycles'
    out['unique'] = {
        'walks': len({r['walk'] for r in uR}), 'motif_graphs': n_graphs(uR),
        'cycles': len({cycle(r['walk']) for r in uR}), 'label_multisets': len({labels(r) for r in uR}),
        'raw_smiles': len({r['raw_smiles'] for r in uR}), 'molecules': len(set(uC)),
        'n': len(uR), 'unique_pct': 100 * len(set(uC)) / len(uR),
        'motifs_per_walk': dict(collections.Counter(len(r['walk'].split('->')) - 1 for r in uR)),
        'novel_pct': 100 * sum(r['novel'] for r in uR) / len(uR),
        # the molecules the original 1000 found and this run does not, and the other way round
        'molecules_gained': len(set(uC) - set(can[:1000])), 'molecules_lost': len(set(can[:1000]) - set(uC)),
        'stats': json.load(open(os.path.join(G, 'unique_n1000_stats.json')))}

json.dump(out, open(os.path.join(G, 'uniqueness.json'), 'w'), indent=1)
L = out['levels']
for n in (1000, 1400):
    print(f"{n:5d} walks: " + ' '.join(f'{k} {L[n][k]}' for k in
          ('walks', 'motif_graphs', 'cycles', 'label_multisets', 'raw_smiles', 'molecules'))
          + f" -> Unique {L[n]['unique_pct']:.0f}%")
print(f"walks 1001-1400 add {out['marginal']['new_molecules']} new molecules "
      f"({out['marginal']['rate_pct']:.0f}% marginal); 1000 distinct would need about "
      f"{out['marginal']['needed_for_1000']} walks")
print(f"{out['duplicate_groups']['n_molecules_from_several_walks']} molecules come from more than one walk, "
      f"{out['duplicate_groups']['rotation_only']} of them from one cycle read at different starting motifs")
if 'unique' in out:
    u = out['unique']
    print(f"unique-cycle {u['n']} walks: " + ' '.join(f'{k} {u[k]}' for k in
          ('walks', 'motif_graphs', 'cycles', 'label_multisets', 'raw_smiles', 'molecules'))
          + f" -> Unique {u['unique_pct']:.0f}% (novel {u['novel_pct']:.0f}%); "
          f"{u['stats']['dup']} duplicate cycles skipped of {u['stats']['closed']} closures examined; "
          f"{u['molecules_gained']} molecules the original 1000 missed, {u['molecules_lost']} it found and this does not")
