"""
Pretrained chemical-language-model embeddings of the 114 dataset monomers -> outputs/<tag>_emb.npz.

polyBERT (kuelumbus/polyBERT) is a DeBERTa trained on 100M hypothetical polymer PSMILES; its sentence-transformers
configuration is mean pooling over the token embeddings, which is what this reproduces with plain transformers.
Caveat worth stating in the write-up: polyBERT expects PSMILES with two [*] connection points, while our dataset
stores each repeat unit as a closed cyclic SMILES, so the strings are slightly out of distribution for it.

polyBERT is loaded from the xushijie/polyBERT mirror (config `_name_or_path: polyBERT_sentence_transformers/`,
hidden size 600, 12 DeBERTa-v2 layers) because the original kuelumbus/polyBERT repo no longer resolves on the Hub.
ChemBERTa-77M-MTR is the canonical DeepChem checkpoint and is included as a second, fully traceable pretrained model.

Run with an environment that has transformers (llamole has 4.47.1):
  ~/miniconda3/envs/llamole/bin/python bl_polybert_embed.py
"""
import os, sys
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
GC = os.path.join(os.path.dirname(H), 'data', 'group-contrib')
MODELS = {'polybert': 'xushijie/polyBERT', 'chemberta': 'DeepChem/ChemBERTa-77M-MTR'}

rows = [l.split() for l in open(os.path.join(GC, 'polymer_walks_v2_smiles.txt')) if l.strip()]
names = [r[-1] for r in rows]
smiles = [r[0] for r in rows]

import torch
from transformers import AutoTokenizer, AutoModel

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
for tag, repo in MODELS.items():
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModel.from_pretrained(repo).eval().to(dev)
    embs = []
    with torch.no_grad():
        for i in range(0, len(smiles), 16):
            batch = tok(smiles[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors='pt').to(dev)
            h = mdl(**batch).last_hidden_state                 # mean pooling, as polyBERT's own ST config does
            m = batch['attention_mask'].unsqueeze(-1).float()
            embs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    E = np.concatenate(embs).astype(np.float32)
    out = os.path.join(H, 'outputs', f'{tag}_emb.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, names=np.array(names), emb=E)
    print('wrote', out, E.shape, 'from', repo, flush=True)
