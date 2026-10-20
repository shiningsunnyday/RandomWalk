"""
Pretrained chemical-language-model embeddings of each motif -> campaign/outputs/motif_emb/<tag>/<i>.npy.

The predictor currently represents every motif by a 2048-bit Morgan fingerprint of the motif molecule (`--mol_feat
fp`). A fingerprint is a hand-designed substructure count; a pretrained chemical language model has instead seen
millions of molecules and encodes the motif in a learned space. Since polyBERT + ridge is the strongest of the
fifteen prediction baselines, the obvious question is whether our architecture improves when its node features come
from that same pretrained space rather than from fingerprints.

main.py already supports this through `--mol_feat dir --mol_feat_dir <dir>`, which loads `<i>.npy` for motif i in
the motif index order, so this script only has to write those files.

Atom-map numbers are stripped before embedding. They mark the motif's R positions, and `mol2fp` ignores them too
(Morgan invariants do not include atom maps), so dropping them keeps the comparison with the fingerprint
like-for-like: both representations see the motif's chemistry and neither sees which site a walk attaches to.

Run with an environment that has transformers (llamole has 4.47.1):
  ~/miniconda3/envs/polymer_walk/bin/python campaign/motif_emb.py --smiles   # step 1, needs utils.graph
  ~/miniconda3/envs/llamole/bin/python    campaign/motif_emb.py             # step 2, needs transformers
"""
import os, sys
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
RV = os.path.dirname(H)
GC = os.path.join(RV, 'data', 'group-contrib')
OUT = os.path.join(H, 'outputs', 'motif_emb')
MODELS = {'polybert': 'xushijie/polyBERT', 'chemberta': 'DeepChem/ChemBERTa-77M-MTR'}

SMI = os.path.join(OUT, 'motif_smiles.txt')


def write_smiles():
    """step 1, in the polymer_walk env: the motifs in main.py's index order, atom maps removed.

    Done as a separate step because the transformers env has no sklearn, which `utils.graph` imports.
    """
    os.environ.setdefault('dataset', GC)
    sys.path.insert(0, os.path.join(RV, 'code'))
    sys.path.insert(0, os.path.join(RV, 'code', 'scripts'))
    from utils.graph import load_mols
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    mols = load_mols(os.path.join(GC, 'all_groups') + '/')
    os.makedirs(OUT, exist_ok=True)
    with open(SMI, 'w') as f:
        for m in mols:
            m = Chem.Mol(m)
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            f.write(Chem.MolToSmiles(m) + '\n')
    print(f'wrote {len(mols)} motif SMILES to {SMI}', flush=True)


def main():
    smiles = [l.strip() for l in open(SMI) if l.strip()]
    print(f'{len(smiles)} motifs; first: {smiles[0]}', flush=True)

    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for tag, repo in MODELS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModel.from_pretrained(repo).eval().to(dev)
        embs = []
        with torch.no_grad():
            for i in range(0, len(smiles), 16):
                b = tok(smiles[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors='pt').to(dev)
                h = mdl(**b).last_hidden_state                      # mean pooling, as polyBERT's own ST config does
                m = b['attention_mask'].unsqueeze(-1).float()
                embs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
        E = np.concatenate(embs).astype(np.float32)
        d = os.path.join(OUT, tag)
        os.makedirs(d, exist_ok=True)
        for i in range(len(smiles)):
            np.save(os.path.join(d, f'{i + 1}.npy'), E[i])
        print(f'wrote {len(smiles)} files of dim {E.shape[1]} to {d} (from {repo})', flush=True)


if __name__ == '__main__':
    (write_smiles if '--smiles' in sys.argv else main)()
