"""Regenerate the 114 SMILES (red_graph.edgelist + all_extra.txt) and instrument every join.

For edge A->B with r_info = {r_grp_1, b1, b2, r_grp_2}:
  A.r_grp_1[i]  <-> B.b2[i]       A's R stand-in replaced by B's real atom      (B's typing is right)
  A.b1[i]       <-> B.r_grp_2[i]  A's real atom replaced by B's R stand-in      (B's typing WINS -> typing bug)
`fix=True` copies A's real-atom typing (element, aromatic, charge, Hs) onto the surviving B stand-in.
"""
import sys, os, json, re, pickle, collections
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networkx as nx
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from reproduce_gas_smiles import _Unpickler, D, ROOT, name_group, load_mols

G = nx.read_edgelist(f'{D}/red_graph.edgelist', create_using=nx.MultiDiGraph)
MOLS = load_mols(f'{D}/all_groups')
DAGS = _Unpickler(open(f'{D}/dags.pkl', 'rb')).load()
LINES = open(os.environ.get('WALKS_FILE', f'{ROOT}/data/polymer_walks_v2_preprocess.txt')).read().splitlines()
_NAMES = open(f'{D}/all_groups/names.txt').read().split() if os.path.exists(f'{D}/all_groups/names.txt') else None
name_group = (lambda m: _NAMES[m-1]) if _NAMES else name_group
NAME_LOOKUP = {name_group(i): i for i in range(1, len(MOLS) + 1)}
# R atoms per group from all_extra.txt (block i <-> i.mol)
_blocks = [b for b in re.split(r'\n\s*\n', open(f'{D}/all_groups/all_extra.txt').read()) if b.strip()]
RED = [{int(l.split(':')[0]) - 1 for l in b.split('\n') if ':' in l} for b in _blocks]
assert len(RED) == len(MOLS)


def strip_unused_r(mol):
    """Delete every R stand-in that no edge consumed (the process_mol convention: open ports of the repeat unit,
    e.g. a terminal S32 keeps its O atoms but not its partner's benzo ring), capping each surviving
    neighbour with one explicit H per removed bond. If that leaves a structure that does not sanitize (an
    unconsumed ring-type R group that shares atoms with the motif's own rings, e.g. S15's benzo ring or the
    S22/L15 bridge carbons), keep the unconsumed R atoms that are ring members and drop only the acyclic ones."""
    out = _strip(mol, keep_ring=False)
    m = Chem.Mol(out)
    try:
        Chem.SanitizeMol(m)
        if len(Chem.GetMolFrags(m)) == 1:
            return out
    except Exception:
        pass
    return _strip(mol, keep_ring=True)


def _strip(mol, keep_ring):
    rw = Chem.RWMol(mol)
    rw.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(rw); ring_info = rw.GetRingInfo()
    drop = []
    for at in rw.GetAtoms():
        tags = at.GetProp('a').split('_')
        node, idx = tags[0].split('-')
        if len(tags) == 1 and int(idx) in RED[NODE_GROUP[node]]:
            if keep_ring and ring_info.NumAtomRings(at.GetIdx()):
                continue
            drop.append(at.GetIdx())
    dropped = set(drop)
    for i in sorted(drop, reverse=True):
        for n in rw.GetAtomWithIdx(i).GetNeighbors():
            if n.GetIdx() in dropped:
                continue
            n.SetNumExplicitHs(n.GetTotalNumHs() + 1); n.SetNoImplicit(True) # keep its existing Hs
        rw.RemoveAtom(i)
    return rw.GetMol()


NODE_GROUP = {}
SOURCE = os.environ.get('SOURCE', 'grps')


def typing(at):
    return (at.GetSymbol(), at.GetIsAromatic(), at.GetFormalCharge(), at.GetTotalNumHs())


def join(mol, other, r1, r2, n_red_a, fix):
    """r1 = A's [r_grp_1 + b1] (composite idx), r2 = B's [b2 + r_grp_2]; first n_red_a pairs are A-red<->B-black."""
    new_mol = Chem.CombineMols(mol, other)
    sep = mol.GetNumAtoms()
    saved = {}
    for i, r_ind in enumerate(r1):
        if i >= n_red_a:  # A black -> replaced by B red stand-in
            a = new_mol.GetAtomWithIdx(r_ind)
            saved[r2[i] + sep] = (a.GetAtomicNum(), a.GetIsAromatic(), a.GetFormalCharge(), a.GetNumExplicitHs(), a.GetNoImplicit())
    ed = Chem.EditableMol(new_mol)
    for i, r_ind in enumerate(r1):
        r = new_mol.GetAtomWithIdx(r_ind)
        ed = Chem.EditableMol(new_mol)
        for n in r.GetNeighbors():
            if n.GetIdx() in r1:
                continue
            bond = new_mol.GetBondBetweenAtoms(r_ind, n.GetIdx())
            ed.RemoveBond(r_ind, n.GetIdx())
            ed.AddBond(n.GetIdx(), r2[i] + sep, order=bond.GetBondType())
            new_mol = ed.GetMol()
    out = Chem.RWMol(new_mol)
    if fix:
        for idx, (z, arom, chg, nh, noimp) in saved.items():
            at = out.GetAtomWithIdx(idx)
            at.SetAtomicNum(z); at.SetIsAromatic(arom); at.SetFormalCharge(chg)
            at.SetNumExplicitHs(nh); at.SetNoImplicit(noimp)
    for r in sorted(r1, reverse=True):
        out.RemoveAtom(r)
    return out.GetMol()


def build(k, fix=False, override=None):
    """override: {(a_id, b_id): e} to try alternative edges."""
    walk = []
    if SOURCE == 'dags_conn':          # original export: dags.pkl conn minus last (loses side chains, see traverse_dfs)
        for a, b, e in DAGS[k][-1][:-1]:
            walk.append([str(a.id), str(b.id), a.val.split(':')[0], b.val.split(':')[0], e])
    else:                              # chain_extract graph (== walks_new/): every walk edge, minus loop-back into root
        g = DAGS[k][0]
        for a, b, x in g.edges(data=True):
            if b == 0:
                continue
            walk.append([str(a), str(b), g.nodes[a]['val'], g.nodes[b]['val'], x['e']])
    for w in walk:
        if override and (w[0], w[1]) in override:
            w[4] = override[(w[0], w[1])]
    edges, idx_mol, vis = collections.defaultdict(list), {}, {}
    for v in walk:
        idx_mol[v[0]], idx_mol[v[1]] = NAME_LOOKUP[v[2]] - 1, NAME_LOOKUP[v[3]] - 1
        vis[v[0]] = vis[v[1]] = False
    NODE_GROUP.clear(); NODE_GROUP.update(idx_mol)
    for v in walk:
        edges[v[0]].append(v[1:])
    a = next(iter(vis)); bfs = [a]; vis[a] = True
    mol = deepcopy(MOLS[idx_mol[a]])
    for i, at in enumerate(mol.GetAtoms()):
        at.SetProp('a', f"{a}-{i}")

    def idxes_of(m):
        d = {}
        for i, at in enumerate(m.GetAtoms()):
            for p in at.GetProp('a').split('_'):
                d[p.split('-')[0], int(p.split('-')[1])] = i
        return d
    idxes = idxes_of(mol)
    records = []
    while bfs:
        a = bfs.pop(0)
        for b, A, B, e in edges[a]:
            if vis[b]:
                continue
            r = G[A][B][e]
            k1 = r['r_grp_1'] + r['b1']; k2 = r['b2'] + r['r_grp_2']
            mol_b = deepcopy(MOLS[idx_mol[b]])
            for i, at in enumerate(mol_b.GetAtoms()):
                at.SetProp('a', f"{b}-{i}")
            ga, gb = MOLS[idx_mol[a]], mol_b
            # typing conflicts where B's stand-in overwrites A's real atom
            conflicts = []
            for ia, ib in zip(r['b1'], r['r_grp_2']):
                ta, tb = typing(ga.GetAtomWithIdx(ia)), typing(gb.GetAtomWithIdx(ib))
                if ta[:2] != tb[:2] or ta[2] != tb[2]:
                    conflicts.append((ia + 1, ta, ib + 1, tb))
            # stand-in vs real on the other side (A's R stand-in vs B's real atom) -- informative only
            stand = []
            for ia, ib in zip(r['r_grp_1'], r['b2']):
                ta, tb = typing(ga.GetAtomWithIdx(ia)), typing(gb.GetAtomWithIdx(ib))
                if ta[0] != tb[0]:
                    stand.append((ia + 1, ta[0], ib + 1, tb[0]))
            records.append(dict(a=a, b=b, A=A, B=B, e=e, r=r, typing_conflicts=conflicts, stand_in_mismatch=stand,
                                n_alt=len(G[A][B])))
            mol = join(mol, mol_b, [idxes[(a, x)] for x in k1], k2, len(r['r_grp_1']), fix)
            for a_ind, b_ind in zip(k1, k2):
                at = mol.GetAtomWithIdx(mol.GetNumAtoms() - mol_b.GetNumAtoms() + b_ind)
                at.SetProp('a', at.GetProp('a') + f'_{a}-{a_ind}')
            idxes = idxes_of(mol); vis[b] = True; bfs.append(b)
    return mol, records


def clean_smiles(mol):
    m = Chem.Mol(mol)
    for at in m.GetAtoms():
        at.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m), True
    except Exception:
        return Chem.MolToSmiles(m), False


if __name__ == '__main__':
    out = []
    for k in range(len(LINES)):
        name = LINES[k].split()[0]
        raw, recs = build(k)
        fixed, _ = build(k, fix=True)
        stripped = strip_unused_r(fixed)
        rs, rok = clean_smiles(raw); fs, fok = clean_smiles(fixed); ss, sok = clean_smiles(stripped)
        out.append(dict(k=k + 1, name=name, walk=LINES[k].split()[1], raw=Chem.MolToSmiles(raw), raw_clean=rs, raw_valid=rok,
                        fixed=fs, fixed_valid=fok, stripped=ss, stripped_valid=sok,
                        edges=[{**{x: y for x, y in r.items() if x != 'r'}, 'r': r['r']} for r in recs]))
        print(f"{k+1:3d} {name:18s} raw_valid={rok} fixed_valid={fok} stripped_valid={sok} typing-conflict edges={sum(bool(r['typing_conflicts']) for r in recs)}/{len(recs)}", flush=True)
    json.dump(out, open(sys.argv[1], 'w'), indent=1, default=str)
