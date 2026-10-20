"""
Dataset, splits, featurizers and metrics for the baseline comparison. Mirrors results_v2/code/main.py exactly, so a
baseline number is directly comparable to the heldout_* columns of a predictor run's metrics.csv:

  * the per-gas-pair mask is `prop[i1] and prop[i2]` over the 114 corrected dataset polymers, kept in file order
    (N = 94 for the H2 and O2 pairs, 92 for the CO2 pair)
  * the split is random.seed(split_seed); shuffle(range(N)); test = the last 20%, validation = the last 20% of the
    remaining 80%, train = the rest. NOTE: the frozen outputs/pred/*/test_idx.txt files are wrong (they were written
    before the idx_partition fix); these splits are recomputed from the seed, which is what the runs actually used.
  * targets are [permeability, selectivity] = [p[i1], p[i1]/p[i2]], normalized with training-split mean/std only
  * metrics are MAE and R^2 on those normalized targets

usage: import bl_data as D; D.pairs(); D.split(pair, seed); D.targets(pair)
"""
import os, re, random, json
from collections import Counter
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
GC = os.path.join(os.path.dirname(H), 'data', 'group-contrib')
WALKS = os.path.join(GC, 'polymer_walks_v2_preprocess.txt')
SMILES = os.path.join(GC, 'polymer_walks_v2_smiles.txt')

PAIRS = {'H2': (0, 1), 'O2': (2, 1), 'CO2': (4, 3)}          # columns of the label tuple are H2 N2 O2 CH4 CO2
TARGET_NAMES = {'H2': ('H2', 'H2/N2'), 'O2': ('O2', 'O2/N2'), 'CO2': ('CO2', 'CO2/CH4')}
SEEDS = (1, 2, 3)
TEST_SIZE = 0.2
VAL_SIZE = 0.2


def _rows():
    out = []
    for line in open(WALKS):
        if not line.strip():
            continue
        tok = line.rstrip('\n').split(' ')
        name, walk, prop = tok[0], tok[1], tok[-1]
        prop = [float(x) if x not in ('-', '_') else None for x in prop.strip('(').rstrip(')').split(',')]
        out.append({'name': name, 'walk': walk, 'prop': prop})
    smi = {l.split()[-1]: l.split()[0] for l in open(SMILES) if l.strip()}
    for r in out:
        r['smiles'] = smi[r['name']]
    return out


ROWS = _rows()
assert len(ROWS) == 114, len(ROWS)


def pairs():
    return list(PAIRS)


def masked(pair):
    """The polymers a predictor run for this gas pair actually sees, in the same order."""
    i1, i2 = PAIRS[pair]
    return [r for r in ROWS if r['prop'][i1] and r['prop'][i2]]


def targets(pair):
    i1, i2 = PAIRS[pair]
    return np.array([[r['prop'][i1], r['prop'][i1] / r['prop'][i2]] for r in masked(pair)], dtype=float)


def split(pair, seed):
    """(train_idx, val_idx, test_idx) as positions into masked(pair)."""
    n = len(masked(pair))
    all_idx = list(range(n))
    random.seed(seed)
    random.shuffle(all_idx)
    k = int(min(1.0, 1 - TEST_SIZE) * n)                      # train_size=1.0 -> 0.8 in idx_partition
    train, test = all_idx[:k], all_idx[k:]
    n_val = int(round(VAL_SIZE * len(train)))
    return train[:-n_val], train[-n_val:], test


def normalize(y, train_idx):
    """Training-split normalization (--train_norm), applied to the whole target matrix."""
    mu, sd = y[train_idx].mean(axis=0), y[train_idx].std(axis=0)
    return (y - mu) / sd, mu, sd


def metrics(y_true, y_pred):
    """MAE and R^2 per target column, on whatever scale is passed in."""
    from sklearn.metrics import r2_score
    return ([float(np.abs(y_pred[:, i] - y_true[:, i]).mean()) for i in range(y_true.shape[1])],
            [float(r2_score(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])])


# ---------------------------------------------------------------- featurizers

def _mols(pair):
    from rdkit import Chem
    return [Chem.MolFromSmiles(r['smiles']) for r in masked(pair)]


_CACHE = {}


def feat(kind, pair):
    """Feature matrix for every polymer in masked(pair). Deterministic and independent of the split."""
    key = (kind, pair)
    if key in _CACHE:
        return _CACHE[key]
    rows = masked(pair)
    if kind == 'morgan':
        from rdkit import Chem
        from rdkit.Chem import AllChem
        X = np.zeros((len(rows), 2048), dtype=np.float32)
        for i, m in enumerate(_mols(pair)):
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            X[i] = np.frombuffer(bytes(fp.ToBitString(), 'ascii'), 'u1') - ord('0')
    elif kind == 'morgan_count':
        from rdkit.Chem import AllChem
        X = np.zeros((len(rows), 2048), dtype=np.float32)
        for i, m in enumerate(_mols(pair)):
            for b, c in AllChem.GetHashedMorganFingerprint(m, 2, nBits=2048).GetNonzeroElements().items():
                X[i, b] = c
    elif kind == 'rdkit_desc':
        from rdkit.Chem import Descriptors
        names = [n for n, _ in Descriptors._descList]
        X = np.zeros((len(rows), len(names)), dtype=np.float64)
        for i, m in enumerate(_mols(pair)):
            X[i] = [f(m) for _, f in Descriptors._descList]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        keep = X.std(axis=0) > 0
        X = X[:, keep]
    elif kind == 'motif_expert':
        # bag of the 99 expert motifs, read straight off the walk string (the labels the grammar uses)
        vocab = sorted({g for r in ROWS for g in re.findall(r'[PSL]\d+', r['walk'])})
        X = np.zeros((len(rows), len(vocab)), dtype=np.float32)
        for i, r in enumerate(rows):
            c = Counter(re.findall(r'[PSL]\d+', r['walk']))
            for g, v in c.items():
                X[i, vocab.index(g)] = v
    elif kind == 'motif_brics':
        # data-driven motif vocabulary: BRICS fragments of the 114 monomers that occur at least twice
        from rdkit import Chem
        from rdkit.Chem import BRICS
        frags = [Counter(Chem.MolToSmiles(Chem.MolFromSmiles(f), canonical=True)
                         for f in BRICS.BRICSDecompose(Chem.MolFromSmiles(r['smiles'])))
                 for r in ROWS]
        tot = Counter()
        for c in frags:
            tot.update(c)
        vocab = sorted([f for f, n in tot.items() if n >= 2])
        by_name = {r['name']: c for r, c in zip(ROWS, frags)}
        X = np.zeros((len(rows), len(vocab)), dtype=np.float32)
        for i, r in enumerate(rows):
            for f, v in by_name[r['name']].items():
                if f in vocab:
                    X[i, vocab.index(f)] = v
    elif kind in ('polybert', 'chemberta'):
        # pretrained chemical language model embeddings, written by bl_polybert_embed.py
        z = np.load(os.path.join(H, 'outputs', f'{kind}_emb.npz'), allow_pickle=True)
        by_name = {n: e for n, e in zip(z['names'].tolist(), z['emb'])}
        X = np.stack([by_name[r['name']] for r in rows]).astype(np.float64)
    else:
        raise ValueError(kind)
    _CACHE[key] = X
    return X


def record(out_path, row):
    """Append one result row to a jsonl file, atomically enough for parallel baseline scripts."""
    import fcntl
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row) + '\n')
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == '__main__':
    for p in pairs():
        tr, va, te = split(p, 1)
        m = masked(p)
        print(f'{p}: N={len(m)} train={len(tr)} val={len(va)} test={len(te)} '
              f'disjoint={len(set(tr) | set(va) | set(te)) == len(m)}')
        print('   targets', targets(p).shape, 'feat sizes',
              {k: feat(k, p).shape[1] for k in ('morgan', 'rdkit_desc', 'motif_expert', 'motif_brics')})
