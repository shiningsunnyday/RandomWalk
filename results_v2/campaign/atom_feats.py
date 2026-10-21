"""
Per-atom pretrained features for every motif, plus the red/black masks the connection-aware predictor needs.

The predictor's node features have always been one 2048-bit Morgan fingerprint per motif, which says what the motif
is but nothing about *how it is being used*: a motif joined through its first R group and the same motif joined
through its second are the same vector. This script produces the raw material for a representation that can tell
them apart — one embedding per atom, so a message can be aggregated over exactly the atoms a given bond consumes.

Encoder: the pretrained GIN of Hu et al. (ICLR 2020, "Strategies for Pre-training Graph Neural Networks"), 5 layers
of 300 dimensions, taken from the checkpoint data_efficient_grammar ships (`supervised_contextpred.pth` by default;
`--ckpt` selects another). `mol_to_graph_data_obj_simple` iterates `mol.GetAtoms()` in order, so row i of the output
is atom i of the .mol file and the red/black masks below index the same rows.

Red and black atoms: `annotate_extra` sets an `r_grp` property on each atom, '' for a black atom and a '-'-separated
list of 1-based red-group numbers otherwise (an atom can belong to more than one). `list_red_members` turns that
into the per-group atom lists, which is what an edge of the motif graph consumes.

usage: PYTHONPATH=$STUBS python campaign/atom_feats.py [--ckpt <name>] [--out <dir>]
  -> campaign/outputs/atom_feats/<name>.npz   with  X (n_atoms, 300), red (n_atoms,) bool, groups (object array of
                                                    per-red-group atom index lists)
"""
import os, sys, argparse
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
RV = os.path.dirname(H)
GC = os.path.join(RV, 'data', 'group-contrib')
DEG = os.path.expanduser('~/data_efficient_grammar')
os.environ.setdefault('dataset', GC)

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='supervised_contextpred.pth')
ap.add_argument('--out', default=os.path.join(H, 'outputs', 'atom_feats'))
cli = ap.parse_args()

sys.path.insert(0, RV)
sys.path.insert(0, os.path.join(RV, 'code'))
from utils.preprocess import load_mols, annotate_extra                               # noqa: E402
from utils.graph import list_red_members                                             # noqa: E402

import torch                                                                         # noqa: E402
sys.path.insert(0, os.path.join(DEG, 'GCN'))
from model import GNN                                                                # noqa: E402
from loader import mol_to_graph_data_obj_simple                                       # noqa: E402


def main():
    mols = load_mols(os.path.join(GC, 'all_groups') + '/')
    annotate_extra(mols, os.path.join(GC, 'all_groups', 'all_extra.txt'))
    print(f'{len(mols)} motifs', flush=True)

    model = GNN(num_layer=5, emb_dim=300, JK='last', drop_ratio=0.0, gnn_type='gin')
    state = torch.load(os.path.join(DEG, 'GCN', 'model_gin', cli.ckpt), map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    # the supervised_* checkpoints carry a task head the node encoder does not use; nothing of the encoder may be
    # missing, or the "pretrained" features would be partly random
    assert not missing, missing
    print(f'loaded {cli.ckpt}; ignored non-encoder keys: {sorted(unexpected)}', flush=True)
    model.eval()

    os.makedirs(cli.out, exist_ok=True)
    tag = cli.ckpt.replace('.pth', '')
    out = {}
    for i, m in enumerate(mols, start=1):
        # the atom map numbers load_mols writes are only for display; strip them so the encoder sees plain atoms
        from rdkit import Chem
        mm = Chem.Mol(m)
        for a in mm.GetAtoms():
            a.SetAtomMapNum(0)
        d = mol_to_graph_data_obj_simple(mm)
        with torch.no_grad():
            X = model(d.x, d.edge_index, d.edge_attr).numpy().astype(np.float32)
        assert X.shape[0] == m.GetNumAtoms(), (i, X.shape, m.GetNumAtoms())
        red = np.array([bool(a.GetProp('r_grp')) for a in m.GetAtoms()])
        groups = list_red_members(m)
        # every atom listed in a red group must be flagged red, and vice versa
        assert set(int(x) for g in groups for x in g) == set(np.flatnonzero(red).tolist()), i
        out[str(i)] = dict(X=X, red=red, groups=np.array(groups, dtype=object))

    np.savez(os.path.join(cli.out, f'{tag}.npz'),
             **{f'{k}_{f}': v[f] for k, v in out.items() for f in ('X', 'red', 'groups')})
    n_red = sum(int(v['red'].sum()) for v in out.values())
    n_at = sum(int(v['X'].shape[0]) for v in out.values())
    print(f'wrote {cli.out}/{tag}.npz: {len(out)} motifs, {n_at} atoms ({n_red} red), dim {X.shape[1]}', flush=True)
    print('red groups per motif:', {k: len(v['groups']) for k, v in list(out.items())[:10]}, '...', flush=True)


if __name__ == '__main__':
    main()
