"""
Train the connection-aware predictor under the paper's protocol and predict with it, so that Table 2 and
everything downstream of the predictor can be re-derived from this model instead of the published one.

What it produces, in the formats the existing pipeline already consumes:

  conn_grid.csv        one row per gas pair x split seed x configuration, with validation and test metrics. Table 2
                       selects the configuration per pair on mean validation loss across the three seeds, exactly as
                       compute_numbers.py does for the published grid.
  *_ensemble.txt       '<walk> <property> <mean> +- <std>' over the three per-seed models, in barrer, which is what
                       ensemble.py writes for the published predictor and what plots.py, apply_si.py and
                       compute_numbers.py read.

Walks arrive in two notations. Dataset walks carry the motif-graph edge at each step (`L3-2>S32`), which is what
the model needs to know which R groups a bond consumes. Generated and experimental walks do not (`L22->S32->L22`),
so the edges are recovered with the grammar's own `verify_walk`, exactly as `gen.py record()` and
`clean_random.py` do, and the structure is therefore the one the paper's SMILES were built from.

usage:
  python campaign/conn_predict.py grid                    -> campaign/outputs/conn/conn_grid.csv
  python campaign/conn_predict.py predict                 -> campaign/outputs/conn/{s9,dataset,fig5}_ensemble.txt
"""
import os, sys, csv, json, argparse, time
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
RV = os.path.dirname(H)
GC = os.path.join(RV, 'data', 'group-contrib')
OUT = os.environ.get('CONN_OUT') or os.path.join(H, 'outputs', 'conn')   # CONN_OUT isolates a variant
os.environ.setdefault('dataset', GC)
sys.path.insert(0, H)
sys.path.insert(0, os.path.join(RV, 'baselines'))
import bl_data as D                                                                  # noqa: E402
import conn_gnn as CG                                                                # noqa: E402
import torch                                                                         # noqa: E402

PAIR_PROPS = {'H2': ('H2', 'H2/N2'), 'O2': ('O2', 'O2/N2'), 'CO2': ('CO2', 'CO2/CH4')}
SEEDS = (1, 2, 3)                      # the paper's three splits; Table 2 reports mean and std over them


# ---------------------------------------------------------------- walks without edge indices

_GRAMMAR = {}


def grammar():
    """the pieces of the generation code needed to re-derive a walk's motif-graph edges"""
    if _GRAMMAR:
        return _GRAMMAR
    import networkx as nx
    sys.path.insert(0, os.path.join(RV, 'code'))
    from diffusion import (load_mols, annotate_extra, r_member_lookup, load_dags,    # noqa: E402
                           DiffusionGraph, verify_walk)
    cfg = json.load(open(os.path.join(RV, 'grammar', 'config.json')))
    cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
    a = argparse.Namespace(**cfg)
    a.dags_file = f'{GC}/dags.pkl'
    a.data_file = a.walks_file = f'{GC}/polymer_walks_v2_preprocess.txt'
    a.motifs_folder = f'{GC}/all_groups/'
    a.extra_label_path = f'{GC}/all_groups/all_extra.txt'
    a.predefined_graph_file = f'{GC}/red_graph.edgelist'
    dargs = {k[len('diffusion_'):]: v for k, v in a.__dict__.items() if 'diffusion' in k}
    gx = nx.read_edgelist(a.predefined_graph_file, create_using=nx.MultiDiGraph)
    mols = load_mols(a.motifs_folder)
    annotate_extra(mols, a.extra_label_path)
    _, dags = load_dags(a)
    _GRAMMAR.update(r_lookup=r_member_lookup(mols), G=DiffusionGraph(dags, gx, **dargs).graph,
                    verify_walk=verify_walk, red=CG.red_graph())
    return _GRAMMAR


def split_walk(w):
    """top-level '->' steps, keeping a bracketed side-chain group with the motif it hangs off, as predict.py does"""
    toks, buf, depth, i = [], '', 0, 0
    while i < len(w):
        if w.startswith('->', i) and depth == 0:
            toks.append(buf); buf = ''; i += 2; continue
        depth += (w[i] == '[') - (w[i] == ']'); buf += w[i]; i += 1
    return toks + [buf]


def graph_from_walk(walk):
    """(nodes, edges) for a walk written without edge indices, via the grammar's own assembly check.

    verify_walk's list form runs the same edge enumeration the generator uses, so the chosen attachment points are
    the ones the structure in Table S8 was built from; its graph form only verifies an assignment already made.
    """
    g = grammar()
    _, conn = g['verify_walk'](g['r_lookup'], g['G'], split_walk(walk), loop_back=True)
    ids, nodes, edges = {}, [], []
    for a, b, e in conn:
        for n in (a, b):
            if n.id not in ids:
                ids[n.id] = len(nodes)
                nodes.append(n.val.split(':')[0])
        edges.append((ids[a.id], ids[b.id], int(e)))
    return nodes, edges


def build_walks(walks, feats, with_edges):
    """feature tensors for a list of walk strings, in order"""
    red = CG.red_graph()
    graphs = []
    for w in walks:
        nodes, edges = CG.parse_walk(w) if with_edges else graph_from_walk(w)
        ids = [CG.IDX[n] for n in nodes]
        E = []
        for u, v, k in edges:
            d = red[nodes[u]][nodes[v]][k]
            E.append((u, v, [int(x) for x in d['r_grp_1']], [int(x) for x in d['r_grp_2']]))
        graphs.append({'ids': ids, 'edges': E})
    return CG.tensors(graphs, feats)


# ---------------------------------------------------------------- training

def fit(bt, bv, y, tr, va, cfg, seed, epochs=400):
    """train one model, returning it at its best validation epoch"""
    torch.manual_seed(seed)
    model = CG.ConnGNN(bt['all'].shape[1], cfg['hid'], cfg['layers'], cfg['dropout'], True,
                       pool=os.environ.get('CONN_POOL', 'mean'))
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    Y = torch.tensor(y, dtype=torch.float32)
    ytr, yva = Y[tr], Y[va]
    best, best_state, best_ep = 1e9, None, -1
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        ((model(bt) - ytr) ** 2).mean().backward(); opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                vl = float(((model(bv) - yva) ** 2).mean())
            if vl < best:
                best, best_ep = vl, ep
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best, best_ep


def grid():
    feats = CG.load_atom_feats()
    rows = []
    t0 = time.time()
    for pair in PAIR_PROPS:
        rowsd = D.masked(pair)
        graphs = build_walks([r['walk'] for r in rowsd], feats, with_edges=True)
        y_raw = D.targets(pair)
        for seed in SEEDS:
            tr, va, te = D.split(pair, seed)
            y, mu, sd = D.normalize(y_raw, tr)
            bt, bv, bte = CG.batch(graphs, tr), CG.batch(graphs, va), CG.batch(graphs, te)
            for ci, cfg in enumerate(CG.GRID):
                model, vl, ep = fit(bt, bv, y, tr, va, cfg, seed)
                with torch.no_grad():
                    tp = model(bte).numpy()
                mae, r2 = D.metrics(y[te], tp)
                mse = [float(((tp[:, i] - y[te][:, i]) ** 2).mean()) for i in range(2)]
                r = dict(pair=pair, seed=seed, config=ci, val_loss=vl, epoch=ep, **cfg)
                for j, p in enumerate(PAIR_PROPS[pair]):
                    r[f'{p}|mae'], r[f'{p}|mse'], r[f'{p}|r^2'] = mae[j], mse[j], r2[j]
                rows.append(r)
            print(f'{pair} seed {seed}: {len(CG.GRID)} configs ({time.time() - t0:.0f}s)', flush=True)
    os.makedirs(OUT, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(OUT, 'conn_grid.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair', 'seed', 'config'] + [k for k in keys
                                                                      if k not in ('pair', 'seed', 'config')])
        w.writeheader(); w.writerows(rows)
    print(f'wrote {OUT}/conn_grid.csv ({len(rows)} rows)')


def chosen_config():
    """the configuration selected per gas pair on mean validation loss across the three seeds"""
    import collections
    rows = list(csv.DictReader(open(os.path.join(OUT, 'conn_grid.csv'))))
    out = {}
    for pair in PAIR_PROPS:
        agg = collections.defaultdict(list)
        for r in rows:
            if r['pair'] == pair:
                agg[int(r['config'])].append(float(r['val_loss']))
        full = {c: np.mean(v) for c, v in agg.items() if len(v) == len(SEEDS)}
        out[pair] = min(full, key=full.get)
    return out


def predict():
    feats = CG.load_atom_feats()
    cfgs = chosen_config()
    sets = {
        'dataset': ([r['walk'] for r in D.ROWS], True),
        's9': ([l.split()[0] for l in open(os.path.join(RV, 'outputs',
                                                        'screening_walks_unique.txt')) if l.strip()], False),
        'fig5': (list(json.load(open(os.path.join(RV, 'inputs', 'experimental_walks.json'))).values()), False),
    }
    # the dataset walk strings carry edge indices; strip them for the output key, which is how every consumer
    # (plots.py, apply_si.py, compute_numbers.py) refers to a walk
    import re as _re
    plain = {'dataset': lambda w: _re.sub(r'-\d+>', '->', w)}

    preds = {k: {} for k in sets}
    for pair in PAIR_PROPS:
        rowsd = D.masked(pair)
        graphs = build_walks([r['walk'] for r in rowsd], feats, with_edges=True)
        y_raw = D.targets(pair)
        targets = {k: build_walks(ws, feats, with_edges=we) for k, (ws, we) in sets.items()}
        cfg = CG.GRID[cfgs[pair]]
        for seed in SEEDS:
            tr, va, te = D.split(pair, seed)
            y, mu, sd = D.normalize(y_raw, tr)
            model, vl, ep = fit(CG.batch(graphs, tr), CG.batch(graphs, va), y, tr, va, cfg, seed)
            for k, (ws, _) in sets.items():
                b = CG.batch(targets[k], list(range(len(ws))))
                with torch.no_grad():
                    p = model(b).numpy() * sd + mu                  # back to barrer and the raw selectivity
                for i, w in enumerate(ws):
                    key = plain.get(k, lambda x: x)(w)
                    for j, prop in enumerate(PAIR_PROPS[pair]):
                        preds[k].setdefault((key, prop), []).append(float(p[i, j]))
            print(f'{pair} seed {seed}: config {cfgs[pair]}, val {vl:.3f}, epoch {ep}', flush=True)

    os.makedirs(OUT, exist_ok=True)
    for k in sets:
        order = []
        for (w, prop) in preds[k]:
            if (w, prop) not in order:
                order.append((w, prop))
        lines = []
        for key in order:
            v = np.array(preds[k][key])
            lines.append(f'{key[0]} {key[1]} {v.mean()} +- {v.std()}')     # population std, as ensemble.py uses
        open(os.path.join(OUT, f'{k}_ensemble.txt'), 'w').write('\n'.join(lines) + '\n')
        print(f'wrote {OUT}/{k}_ensemble.txt ({len(lines)} lines)')
    json.dump({p: {'config': cfgs[p], **CG.GRID[cfgs[p]]} for p in cfgs},
              open(os.path.join(OUT, 'selected_configs.json'), 'w'), indent=1)


if __name__ == '__main__':
    {'grid': grid, 'predict': predict}[sys.argv[1]]()
