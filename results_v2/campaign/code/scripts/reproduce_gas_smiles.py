"""
Independently regenerate datasets/group-contrib-smiles/smiles_and_props_{H2_N2,O2_N2,CO2_CH4}.txt
from the source of truth, and compare against the shipped files.

Pipeline (reconstructed; the original writer script is not in either repo):
  1. data/polymer_walks_v2_preprocess.txt   walk strings with exact edge ids, e.g. L3-2>S32-2>S20[-1>S1,-4>S1]-2>S32-2>L3
     -> utils.graph.chain_extract + verify_walk (construct_graph.py) -> data/datasets/group-contrib/dags.pkl
        (per polymer: (DiGraph, root Node, conn=[(Node a, Node b, e), ...]); last conn entry is the loop-back edge)
  2. RandomWalk main.py export: conn[:-1] (drop loop-back), write_conn -> (a.id, b.id, A, B, str(e), G[A][B][e])
  3. 3D-motif-grammar src/utils/walk.py walk_to_mol: BFS from the first node over forward edges; each edge's
     r_info = {r_grp_1, b1, b2, r_grp_2} says: delete A's atoms (r_grp_1 + b1) and re-point their bonds to B's
     atoms (b2 + r_grp_2).  join_mols of that era: B's element/H-count always wins, no sanitize.
     -> Chem.MolToSmiles  (written as smiles,prop1,prop2 into data/datasets/group-contrib/smiles_and_props.txt)
  4. Gas-pair files: polymers whose (prop[i1], prop[i2]) are both present, columns  smiles  prop[i1]  prop[i1]/prop[i2]
     with props = (H2, N2, O2, CH4, CO2):  H2_N2=(0,1)  O2_N2=(2,1)  CO2_CH4=(4,3)    [main.py preprocess_data]

Uses the Jan-2024 artifacts (motif graph data/backup_2026-10-07/red_graph.edgelist, the original 97 groups
from data/backup_2026-10-15_pre_group_overhaul/all_groups): dags.pkl / walks as they were before the 2026-10-06 walk-label fixes
(backed up under data/backup_2026-10-06/).

usage (no utils/moses needed):
  python scripts/reproduce_gas_smiles.py [--graph red_graph.edgelist] [--out_dir DIR]
"""
import argparse, os, pickle, sys
from collections import defaultdict
from copy import deepcopy
import networkx as nx
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get('GC_DIR', os.path.join(ROOT, 'data/datasets/group-contrib'))
PAIRS = {'H2_N2': (0, 1), 'O2_N2': (2, 1), 'CO2_CH4': (4, 3)}


class _Node:  # stand-in for utils.graph.Node so dags.pkl loads without importing utils (moses, eden, ...)
    pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'utils.graph' and name == 'Node':
            return _Node
        return super().find_class(module, name)


def name_group(m):
    return ("P" if m <= 41 else ("S" if m <= 73 else "L")) + f"{m if m<=41 else m-41 if m<=73 else m-73}"


def load_mols(folder):
    """3D-motif-grammar load_mols: plain MolFromMolFile, no atom-map numbering added."""
    files = sorted((f for f in os.listdir(folder) if f.endswith('.mol')), key=lambda x: int(x[:-4]))
    return [Chem.MolFromMolFile(os.path.join(folder, f)) or
            Chem.MolFromMolFile(os.path.join(folder, f), sanitize=False) for f in files]


def join_mols(mol, other, r1, r2):
    """Jan-2024 join (3D rigid alignment omitted; it only moves coordinates)."""
    new_mol = Chem.CombineMols(mol, other)
    sep = mol.GetNumAtoms()
    ed_new = Chem.EditableMol(new_mol)
    for i, r_ind in enumerate(r1):
        r = new_mol.GetAtomWithIdx(r_ind)
        ed_new = Chem.EditableMol(new_mol)
        for n in r.GetNeighbors():
            if n.GetIdx() in r1:
                continue
            bond = new_mol.GetBondBetweenAtoms(r_ind, n.GetIdx())
            ed_new.RemoveBond(r_ind, n.GetIdx())
            ed_new.AddBond(n.GetIdx(), r2[i] + sep, order=bond.GetBondType())
            new_mol = ed_new.GetMol()
    for r in sorted(r1, reverse=True):
        ed_new.RemoveAtom(r)
    return ed_new.GetMol()


def walk_to_mol(walk, mols):
    name_lookup = {name_group(i): i for i in range(1, len(mols) + 1)}

    def extract_idxes(mol):
        dic = {}
        for i, a in enumerate(mol.GetAtoms()):
            for p in a.GetProp('a').split('_'):
                dic[p.split('-')[0], int(p.split('-')[1])] = i
        return dic

    edges, idx_mol, vis = defaultdict(list), {}, {}
    for v in walk:
        idx_mol[v[0]], idx_mol[v[1]] = name_lookup[v[2]] - 1, name_lookup[v[3]] - 1
        vis[v[0]] = vis[v[1]] = False
    for v in walk:
        edges[v[0]].append(v[1:])
    a = next(iter(vis))
    bfs, vis[a] = [a], True
    mol = deepcopy(mols[idx_mol[a]])
    for i, at in enumerate(mol.GetAtoms()):
        at.SetProp('a', f"{a}-{i}")
    idxes = extract_idxes(mol)
    while bfs:
        a = bfs.pop(0)
        for b, _, _, _, r_info in edges[a]:
            if vis[b]:
                continue
            k1 = r_info['r_grp_1'] + r_info['b1']
            k2 = r_info['b2'] + r_info['r_grp_2']
            mol_b = deepcopy(mols[idx_mol[b]])
            for i, at in enumerate(mol_b.GetAtoms()):
                at.SetProp('a', f"{b}-{i}")
            mol = join_mols(mol, mol_b, [idxes[(a, r)] for r in k1], k2)
            for a_ind, b_ind in zip(k1, k2):
                at = mol.GetAtomWithIdx(mol.GetNumAtoms() - mol_b.GetNumAtoms() + b_ind)
                at.SetProp('a', at.GetProp('a') + f'_{a}-{a_ind}')
            idxes = extract_idxes(mol)
            vis[b] = True
            bfs.append(b)
    return mol


def props_of(line):
    p = line.split()[-1].strip('()').split(',')
    return [float(x) if x not in ('-', '_') else None for x in p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', default=os.path.join(ROOT, 'data/backup_2026-10-07/red_graph.edgelist'),
                    help='edgelist in data/datasets/group-contrib; dags.pkl/walks_new edge ids e index THIS graph '
                         '(829/829 edges consistent), not red_graph_new_s21.edgelist (578/829)')
    ap.add_argument('--out_dir', default=None, help='where to write the regenerated files (default: print stats only)')
    args = ap.parse_args()

    G = nx.read_edgelist(args.graph, create_using=nx.MultiDiGraph)
    # the 97 original groups (the .mol files are unchanged since Jan 2024 in this snapshot)
    mols = load_mols(os.path.join(ROOT, 'data/backup_2026-10-15_pre_group_overhaul/all_groups'))
    B = os.path.join(ROOT, 'data/backup_2026-10-06')
    dags = _Unpickler(open(os.path.join(B, 'dags.pkl'), 'rb')).load()
    lines = open(os.path.join(B, 'polymer_walks_v2_preprocess.txt')).read().splitlines()

    smiles = []
    for k in range(len(lines)):
        walk = []
        for a, b, e in dags[k][-1][:-1]:                # main.py: everything except last loop back
            A, B = a.val.split(':')[0], b.val.split(':')[0]
            walk.append((str(a.id), str(b.id), A, B, str(e), G[A][B][e]))
        smiles.append(Chem.MolToSmiles(walk_to_mol(walk, mols)))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    for tag, (i1, i2) in PAIRS.items():
        rows = []
        for k, l in enumerate(lines):
            p = props_of(l)
            if p[i1] and p[i2]:
                rows.append(f"{smiles[k]} {p[i1]} {p[i1] / p[i2]}")
        ref = open(os.path.join(ROOT, f'datasets/group-contrib-smiles/smiles_and_props_{tag}.txt')).read().splitlines()
        same_props = len(rows) == len(ref) and all(r.split()[1:] == s.split()[1:] for r, s in zip(rows, ref))
        same_smi = sum(r.split()[0] == s.split()[0] for r, s in zip(rows, ref))
        print(f"{tag:8s} rows {len(rows)}/{len(ref)}  props identical: {same_props}  SMILES identical: {same_smi}/{len(ref)}")
        if args.out_dir:
            open(os.path.join(args.out_dir, f'smiles_and_props_{tag}.txt'), 'w').write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
