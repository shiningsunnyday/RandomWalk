"""
Connection-aware predictor: message passing over the walk's own DAG, with messages that depend on which R group of
the sending motif the bond consumes.

Why. In the published predictor a motif is one 2048-bit Morgan fingerprint, and that vector is the same whichever
of the motif's R groups is being used. Two polymers that join the same two motifs through different attachment
points are therefore indistinguishable at the node level, and differ only through the grammar's transition weights.
The red graph already records the exact choice for every bond — `{'r_grp_1': [...], 'b1': [...], 'b2': [...],
'r_grp_2': [...]}`, atom indices into the two motifs — so the information is there; nothing in the model reads it.

Model. Per-atom pretrained embeddings (campaign/atom_feats.py, a pretrained GIN) give one vector per atom of every
motif. Then, for a walk:

  h_v^0        = MLP_in( mean over ALL atoms of motif(v) )                       whole group, red atoms included
  m_{u->v}     = MLP_base([h_u ; mean over u's BLACK atoms])
               + MLP_red ([h_u ; mean over the RED atoms of u that this bond consumes])
  h_v^{l+1}    = h_v^l + ReLU( MLP_upd([h_v^l ; sum_u m_{u->v}]) )
  y            = MLP_out( mean_v h_v^L )

The red term is what makes a message orientation-dependent: u->v consumes `r_grp_1` of u, v->u consumes `r_grp_2`
of v, and a different bond between the same pair of motifs consumes different atoms. `--red all` replaces it with
the mean over all of u's red atoms, which is the ablation that removes exactly the connection-awareness and keeps
everything else; `--msg base` drops the red term entirely; `--feat fp` swaps the pretrained per-atom features for
the published Morgan fingerprint (no red/black distinction available, so the red term uses the same vector).

Protocol is bl_data's, which mirrors main.py: per gas pair, 80/20 split from the seed, 20% of train as validation,
training-split normalization, validation alone picks the epoch and the config, MAE/R^2 on normalized targets.

usage: python campaign/conn_gnn.py --arm conn --seeds 1-10 [--feat atom|fp] [--msg both|base] [--red edge|all]
  -> campaign/outputs/conn/<arm>.csv  (one row per pair x seed x config) and a printed summary
"""
import os, sys, re, json, time, argparse
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
RV = os.path.dirname(H)
GC = os.path.join(RV, 'data', 'group-contrib')
sys.path.insert(0, os.path.join(RV, 'baselines'))
import bl_data as D                                                                  # noqa: E402

import torch                                                                         # noqa: E402
import torch.nn as nn                                                                # noqa: E402

# these graphs have tens of nodes; torch's default thread pool costs far more than it saves and several arms run
# side by side, so one thread per process is both faster and neighbourly
torch.set_num_threads(1)

ap = argparse.ArgumentParser()
ap.add_argument('--arm', default='conn')
ap.add_argument('--seeds', default='1-10')
ap.add_argument('--feat', default='atom', choices=['atom', 'fp'])
ap.add_argument('--msg', default='both', choices=['both', 'base'])
ap.add_argument('--red', default='edge', choices=['edge', 'all'])
ap.add_argument('--atom_npz', default=os.path.join(H, 'outputs', 'atom_feats', 'supervised_contextpred.npz'))
ap.add_argument('--epochs', type=int, default=400)
# the two campaign changes that are orthogonal to the representation (BRIEFING_2 6b): a robust loss and the log
# target. --log_target takes log10 of permeability and selectivity before the training-split normalization, exactly
# as campaign/code/main.py's --log_target does, so this arm is comparable to the campaign's `log` arm and not to
# `base`, whose MAE is in raw-normalized units.
ap.add_argument('--huber', type=float, default=0.0)
ap.add_argument('--log_target', action='store_true')
ap.add_argument('--out', default=os.path.join(H, 'outputs', 'conn'))
cli = ap.parse_args()
SEEDS = ([int(x) for x in cli.seeds.split(',')] if ',' in cli.seeds else
         list(range(int(cli.seeds.split('-')[0]), int(cli.seeds.split('-')[1]) + 1)))

# ---------------------------------------------------------------- motifs, atoms and the red graph

NAMES = [l.strip() for l in open(os.path.join(GC, 'all_groups', 'names.txt')) if l.strip()]
IDX = {n: i + 1 for i, n in enumerate(NAMES)}                      # motif name -> 1-based index, as in main.py


def load_atom_feats():
    z = np.load(cli.atom_npz, allow_pickle=True)
    out = {}
    for i in range(1, len(NAMES) + 1):
        X = z[f'{i}_X']; red = z[f'{i}_red']; groups = z[f'{i}_groups']
        out[i] = {'X': X, 'red': red, 'groups': [list(map(int, g)) for g in groups]}
    return out


def load_fp_feats():
    """the published node feature, so the architecture can be compared at equal information about the motif"""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    sys.path.insert(0, RV)
    from utils.preprocess import load_mols
    mols = load_mols(os.path.join(GC, 'all_groups') + '/')
    out = {}
    for i, m in enumerate(mols, start=1):
        mm = Chem.Mol(m)
        for a in mm.GetAtoms():
            a.SetAtomMapNum(0)
        fp = AllChem.GetMorganFingerprintAsBitVect(mm, 2, nBits=2048)
        v = (np.frombuffer(bytes(fp.ToBitString(), 'ascii'), 'u1') - ord('0')).astype(np.float32)
        # one vector per motif, repeated for every atom: identical information to --mol_feat fp, so any difference
        # against --feat atom is the per-atom encoder and not the architecture
        out[i] = {'X': np.tile(v, (m.GetNumAtoms(), 1)), 'red': np.zeros(m.GetNumAtoms(), bool),
                  'groups': [list(range(m.GetNumAtoms()))]}
    return out


def red_graph():
    import networkx as nx
    return nx.read_edgelist(os.path.join(GC, 'red_graph.edgelist'), create_using=nx.MultiDiGraph)


TOK = re.compile(r'([PSL]\d+)|(-(\d+)>)|(\[)|(\])|(,)')


def parse_walk(s):
    """(nodes, edges) for a dataset walk string such as 'L3-2>S32-2>S20[-1>S1,-4>S1]-2>S32-2>L3'.

    nodes[i] is a motif name; edges are (u, v, key) with u, v positions into nodes. The walk is closed, so the
    final motif is the first one read a second time and the two are merged into one node.
    """
    toks = [m.group(0) for m in TOK.finditer(s)]
    assert ''.join(toks) == s, (s, ''.join(toks))
    nodes, edges = [], []
    pos = 0

    def motif():
        nonlocal pos
        assert re.fullmatch(r'[PSL]\d+', toks[pos]), (s, toks[pos])
        nodes.append(toks[pos]); pos += 1
        return len(nodes) - 1

    def chain(prev):
        """prev -e> motif, then that motif's own side chains and continuation"""
        nonlocal pos
        while pos < len(toks) and toks[pos].startswith('-'):
            key = int(toks[pos][1:-1]); pos += 1
            cur = motif()
            edges.append((prev, cur, key))
            if pos < len(toks) and toks[pos] == '[':
                pos += 1
                while True:
                    chain(cur)
                    if toks[pos] == ',':
                        pos += 1
                        continue
                    assert toks[pos] == ']', (s, toks[pos])
                    pos += 1
                    break
            prev = cur
            if pos < len(toks) and toks[pos] in ',]':
                return
        return

    first = motif()
    if pos < len(toks) and toks[pos] == '[':
        pos += 1
        while True:
            chain(first)
            if toks[pos] == ',':
                pos += 1
                continue
            assert toks[pos] == ']', (s, toks[pos])
            pos += 1
            break
    chain(first)
    assert pos == len(toks), (s, toks[pos:])
    assert nodes[-1] == nodes[0], (s, nodes)               # closed walk
    last = len(nodes) - 1
    nodes = nodes[:-1]
    edges = [(first if u == last else u, first if v == last else v, k) for u, v, k in edges]
    return nodes, edges


def build(rows, feats, G):
    """one graph per polymer: node motif ids, and per-edge the atom-index sets the bond consumes"""
    out = []
    for r in rows:
        nodes, edges = parse_walk(r['walk'])
        ids = [IDX[n] for n in nodes]
        E = []
        for u, v, k in edges:
            a, b = nodes[u], nodes[v]
            d = G[a][b][k]
            E.append((u, v, [int(x) for x in d['r_grp_1']], [int(x) for x in d['r_grp_2']]))
        out.append({'ids': ids, 'edges': E})
    return out


def pooled(feats, mid, which, idx=None):
    """mean of a motif's per-atom features over all / black / a given atom subset"""
    f = feats[mid]
    if which == 'all':
        return f['X'].mean(0)
    if which == 'black':
        m = ~f['red']
        return f['X'][m].mean(0) if m.any() else f['X'].mean(0)
    sel = [i for i in idx if i < f['X'].shape[0]]
    return f['X'][sel].mean(0) if sel else f['X'].mean(0)


def tensors(graphs, feats):
    """static per-node and per-edge vectors, computed once"""
    out = []
    for g in graphs:
        ids = g['ids']
        node_all = np.stack([pooled(feats, i, 'all') for i in ids])
        node_black = np.stack([pooled(feats, i, 'black') for i in ids])
        src, dst, red = [], [], []
        for (u, v, r1, r2) in g['edges']:
            for a, b, ridx in ((u, v, r1), (v, u, r2)):          # both orientations, each with its own red group
                src.append(a); dst.append(b)
                if cli.red == 'all':
                    f = feats[ids[a]]
                    red.append(f['X'][f['red']].mean(0) if f['red'].any() else f['X'].mean(0))
                else:
                    red.append(pooled(feats, ids[a], 'idx', ridx))
        out.append({'all': node_all.astype(np.float32), 'black': node_black.astype(np.float32),
                    'src': np.array(src, dtype=np.int64), 'dst': np.array(dst, dtype=np.int64),
                    'red': (np.stack(red).astype(np.float32) if red
                            else np.zeros((0, node_all.shape[1]), np.float32))})
    return out


def batch(graphs, idx):
    """one disjoint-union graph for a whole split: the model then costs a handful of tensor ops per epoch rather
    than one forward pass per polymer, which is the difference between minutes and seconds per run"""
    off, A, B, S, Dd, R, gid = 0, [], [], [], [], [], []
    for gi, i in enumerate(idx):
        g = graphs[i]
        n = g['all'].shape[0]
        A.append(g['all']); B.append(g['black']); R.append(g['red'])
        S.append(g['src'] + off); Dd.append(g['dst'] + off)
        gid.append(np.full(n, gi, dtype=np.int64))
        off += n
    return {'all': torch.from_numpy(np.concatenate(A)), 'black': torch.from_numpy(np.concatenate(B)),
            'red': torch.from_numpy(np.concatenate(R)), 'src': torch.from_numpy(np.concatenate(S)),
            'dst': torch.from_numpy(np.concatenate(Dd)), 'gid': torch.from_numpy(np.concatenate(gid)),
            'n_graphs': len(idx)}


# ---------------------------------------------------------------- model

class ConnGNN(nn.Module):
    def __init__(self, in_dim, hid, layers, dropout, use_red):
        super().__init__()
        self.use_red = use_red
        self.inp = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.base_proj = nn.Linear(in_dim, hid)
        self.red_proj = nn.Linear(in_dim, hid)
        self.base = nn.ModuleList(nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU()) for _ in range(layers))
        self.red = nn.ModuleList(nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU()) for _ in range(layers))
        self.upd = nn.ModuleList(nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Dropout(dropout),
                                               nn.Linear(hid, hid)) for _ in range(layers))
        self.out = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, 2))

    def forward(self, g):
        h = self.inp(g['all'])
        base = self.base_proj(g['black'])
        red = self.red_proj(g['red']) if g['red'].shape[0] else None
        for bl, rl, ul in zip(self.base, self.red, self.upd):
            if g['src'].shape[0]:
                m = bl(torch.cat([h[g['src']], base[g['src']]], -1))
                if self.use_red and red is not None:
                    m = m + rl(torch.cat([h[g['src']], red], -1))
                agg = torch.zeros_like(h).index_add_(0, g['dst'], m)
            else:
                agg = torch.zeros_like(h)
            h = h + ul(torch.cat([h, agg], -1))
        # mean over each polymer's own nodes
        s = torch.zeros((g['n_graphs'], h.shape[1]), dtype=h.dtype).index_add_(0, g['gid'], h)
        c = torch.zeros((g['n_graphs'], 1), dtype=h.dtype).index_add_(
            0, g['gid'], torch.ones((h.shape[0], 1), dtype=h.dtype))
        return self.out(s / c)


GRID = [dict(hid=h, layers=L, dropout=d, lr=lr)
        for h in (64, 128) for L in (2, 3) for d in (0.0, 0.2) for lr in (1e-3,)]
if os.environ.get('CONN_FIXED'):
    # one fixed configuration, no search: the published model selects over two settings per arm, so a comparison
    # against it with an eight-point grid gives this architecture a larger search. This is the modal selection
    # (hid 128, 3 layers, no dropout); with it the only thing validation still chooses is the epoch.
    GRID = [dict(hid=128, layers=3, dropout=0.0, lr=1e-3)]


def run_one(bt, bv, bte, y, tr, va, te, cfg, seed):
    torch.manual_seed(seed)
    model = ConnGNN(bt['all'].shape[1], cfg['hid'], cfg['layers'], cfg['dropout'], cli.msg == 'both')
    lossf = nn.HuberLoss(delta=cli.huber) if cli.huber > 0 else None
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    Y = torch.tensor(y, dtype=torch.float32)
    ytr, yva = Y[tr], Y[va]
    best = (1e9, None, -1)
    for ep in range(cli.epochs):
        model.train(); opt.zero_grad()
        loss = (lossf(model(bt), ytr) if lossf is not None else ((model(bt) - ytr) ** 2).mean())
        loss.backward(); opt.step()
        if ep % 5 == 0 or ep == cli.epochs - 1:
            model.eval()
            with torch.no_grad():
                vl = float(((model(bv) - yva) ** 2).mean())
                if vl < best[0]:
                    best = (vl, model(bte).numpy(), ep)
    return best


def main():
    feats = load_atom_feats() if cli.feat == 'atom' else load_fp_feats()
    G = red_graph()
    os.makedirs(cli.out, exist_ok=True)
    rows_out = []
    t0 = time.time()
    for pair in D.pairs():
        rows = D.masked(pair)
        graphs = tensors(build(rows, feats, G), feats)
        y_true_raw = D.targets(pair)                 # always the raw targets, for the comparable metric below
        y_raw = y_true_raw
        if cli.log_target:
            assert (y_raw > 0).all(), 'log_target needs strictly positive targets'
            y_raw = np.log10(y_raw)
        for seed in SEEDS:
            tr, va, te = D.split(pair, seed)
            y, mu, sd = D.normalize(y_raw, tr)
            bt, bv, bte = batch(graphs, tr), batch(graphs, va), batch(graphs, te)
            picked = None
            for ci, cfg in enumerate(GRID):
                vl, tp, ep = run_one(bt, bv, bte, y, tr, va, te, cfg, seed)
                if picked is None or vl < picked[0]:
                    picked = (vl, tp, ep, ci, cfg)
            vl, tp, ep, ci, cfg = picked                     # validation alone picks config and epoch
            mae, r2 = D.metrics(y[te], tp)
            # A log-target model's MAE and R^2 live on log-transformed targets, which is a different and easier
            # metric: it is not comparable with a raw-target arm, nor with the baselines, which back-transform.
            # So report both. The raw-scale numbers invert the log, renormalize with the training split's raw
            # statistics and score against the raw truth, exactly as baselines/bl_classical.py's fit_eval does.
            if cli.log_target:
                mu_l, sd_l = y_raw[tr].mean(axis=0), y_raw[tr].std(axis=0)
                mu_r, sd_r = y_true_raw[tr].mean(axis=0), y_true_raw[tr].std(axis=0)
                p_raw = (np.power(10.0, np.clip(tp * sd_l + mu_l, -8, 9)) - mu_r) / sd_r
                mae_raw, r2_raw = D.metrics((y_true_raw[te] - mu_r) / sd_r, p_raw)
            else:
                mae_raw, r2_raw = mae, r2
            for j, name in enumerate(D.TARGET_NAMES[pair]):
                rows_out.append(dict(arm=cli.arm, pair=pair, seed=seed, prop=name, mae=mae[j], r2=r2[j],
                                     raw_mae=mae_raw[j], raw_r2=r2_raw[j],
                                     val_loss=vl, epoch=ep, config=ci, **cfg))
            print(f'{pair} seed {seed}: val {vl:.3f} ep {ep} cfg {ci} '
                  f'MAE {mae[0]:.3f}/{mae[1]:.3f} R2 {r2[0]:.3f}/{r2[1]:.3f} ({time.time() - t0:.0f}s)', flush=True)
    import csv
    path = os.path.join(cli.out, f'{cli.arm}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    mae = float(np.mean([r['mae'] for r in rows_out])); r2 = float(np.mean([r['r2'] for r in rows_out]))
    mae_r = float(np.mean([r['raw_mae'] for r in rows_out])); r2_r = float(np.mean([r['raw_r2'] for r in rows_out]))
    V_r = float(np.mean([np.std([r['raw_r2'] for r in rows_out if r['prop'] == p])
                         for p in {r['prop'] for r in rows_out}]))
    print(f'{cli.arm}: back-transformed raw scale  MAE {mae_r:.3f}  R2 {r2_r:.3f}  V {V_r:.3f}')
    V = float(np.mean([np.std([r['r2'] for r in rows_out if r['prop'] == p])
                       for p in {r['prop'] for r in rows_out}]))
    print(f'\n{cli.arm}: MAE {mae:.3f}  R2 {r2:.3f}  V {V:.3f}  ({len(rows_out)} cells, {time.time() - t0:.0f}s)')
    json.dump({'arm': cli.arm, 'feat': cli.feat, 'msg': cli.msg, 'red': cli.red,
               'mae': mae, 'r2': r2, 'V': V, 'n': len(rows_out),
               'raw_mae': mae_r, 'raw_r2': r2_r, 'raw_V': V_r, 'log_target': bool(cli.log_target)},
              open(os.path.join(cli.out, f'{cli.arm}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
