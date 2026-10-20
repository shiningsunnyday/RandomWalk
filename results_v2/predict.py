"""
Predict permeability/selectivity of walks with trained v2 predictor runs (results_v2 code/data/grammar).

usage: python results_v2/predict.py --walks <walks.txt> --out <members.csv> <predictor_dir> [...]
  walks.txt: one walk per line, '->'-separated motifs with bracketed side chains (e.g. L5->S21[->P17,->P17]->L5);
             only the first whitespace-separated token of each line is used.
Each predictor_dir is used at its saved checkpoint with the lowest validation loss (predictor_ckpt_<val loss>.pt), and
predictions are de-normalized with that run's training-split statistics (mean_and_std.txt written at training time).
The predictor input depends on the motif sequence only (motif fingerprints, grammar transition weights), not on which
motif-graph edge joins two motifs, so regio-isomers that share a motif sequence get identical predictions.
"""
import os, sys, json, glob, argparse, csv
H = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--walks'); ap.add_argument('--dataset', action='store_true', help='predict the 114 dataset polymers from their stored DAGs (as in training)'); ap.add_argument('--out', required=True); ap.add_argument('pdirs', nargs='+')
cli = ap.parse_args()
out_csv = os.path.abspath(cli.out); walks_path = os.path.abspath(cli.walks) if cli.walks else None; pdirs = [os.path.abspath(p) for p in cli.pdirs]
GC = os.path.join(H, 'data', 'group-contrib')
os.environ['dataset'] = GC
sys.path.insert(0, os.path.join(H, 'code')); os.chdir(os.path.join(H, 'code'))
import numpy as np, torch, networkx as nx
from main import *  # noqa
from utils.graph import is_novel, dfs_traverse

args = argparse.Namespace(motifs_folder=GC + '/all_groups/', extra_label_path=GC + '/all_groups/all_extra.txt',
    grammar_folder=os.path.join(H, 'grammar'), predefined_graph_file=GC + '/red_graph.edgelist',
    dags_file=GC + '/dags.pkl', walks_file=GC + '/polymer_walks_v2_preprocess.txt',
    mol_feat=['fp'], feat_concat_W=True, concat_mol_feats=False, cuda=-1, task='regression',
    ablate_bidir=False, attn_W=False, edge_weights=False)
data, dags = load_dags(args)
cfg_g = json.load(open(os.path.join(args.grammar_folder, 'config.json'))); cfg_g = json.loads(cfg_g) if isinstance(cfg_g, str) else cfg_g
diffusion_args = {k[len('diffusion_'):]: v for (k, v) in cfg_g.items() if 'diffusion' in k}
graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
graph = DiffusionGraph(dags, graph, **diffusion_args)
G = graph.graph
mols = load_mols(args.motifs_folder)
annotate_extra(mols, args.extra_label_path)
feat_lookup = {name_group(i): mol2fp(mols[i - 1]) for i in range(1, len(mols) + 1)}
mol_feats = np.zeros((len(G), len(feat_lookup[list(feat_lookup)[0]])), dtype=np.float32)
for n in G.nodes():
    mol_feats[graph.index_lookup[n]] = feat_lookup[n.split(':')[0]]
diffusion_args['adj_matrix'] = nx.adjacency_matrix(G).toarray()
model = L_grammar(len(G), diffusion_args)
model.load_state_dict(torch.load(os.path.join(args.grammar_folder, 'ckpt.pt')))
model.eval()
if cli.dataset:
    import re as _re
    _lines = open(args.walks_file).read().split('\n')
    walks = [_re.sub(r'-\d+>', '->', l.split()[1]) for l in _lines if l.strip()]
    dag_of = {w: d for w, d in zip(walks, sorted(dags, key=lambda d: d.dag_id))}
else:
    walks = [l.split()[0] for l in open(walks_path) if l.strip()]
    dag_of = {}
print('N', len(G), 'input_dim', mol_feats.shape[1] + len(G), flush=True)


def split_walk(w):
    toks, buf, depth, i = [], '', 0, 0
    while i < len(w):
        if w.startswith('->', i) and depth == 0:
            toks.append(buf); buf = ''; i += 2; continue
        depth += (w[i] == '[') - (w[i] == ']'); buf += w[i]; i += 1
    return toks + [buf]


def tuplify(n):
    n.children = [c if isinstance(c, tuple) else (c, 0) for c in n.children]
    if n.parent is not None and not isinstance(n.parent, tuple): n.parent = (n.parent, 0)
    for c in n.children: tuplify(c[0])


cache = {}
def features(w, directed):
    if (w, directed) not in cache and w in dag_of:            # dataset polymer: exactly the training representation
        root = dag_of[w]; proc = graph.lookup_process(root.dag_id)
        a2 = argparse.Namespace(**{**vars(args), 'ablate_bidir': directed})
        cache[(w, directed)] = (0,) + tuple(featurize_walk(a2, graph, model, root, proc, mol_feats, feat_lookup))
    if (w, directed) not in cache:
        root, conn = dfs_traverse(split_walk(w), loop_back=True)
        for a, b in conn:
            assert b.val in G[a.val], (w, a.val, b.val)
        tuplify(root)
        DiffusionGraph.value_count(root, {})
        novel = is_novel(deepcopy(dags), root)
        proc = DiffusionProcess(root, graph.index_lookup, **diffusion_args)
        a2 = argparse.Namespace(**{**vars(args), 'ablate_bidir': directed})
        cache[(w, directed)] = (novel,) + tuple(featurize_walk(a2, graph, model, root, proc, mol_feats, feat_lookup))
    return cache[(w, directed)]


names = ['H2', 'N2', 'O2', 'CH4', 'CO2']
rows = []
for pd_ in pdirs:
    cfg = json.load(open(os.path.join(pd_, 'config.json'))); cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
    ck = min(glob.glob(os.path.join(pd_, 'predictor_ckpt_*.pt')), key=lambda p: float(os.path.basename(p)[15:-3]))
    ms = np.atleast_2d(np.loadtxt(os.path.join(pd_, 'mean_and_std.txt'))); mean, std = ms[:, 0], ms[:, 1]
    predictor = Predictor(num_heads=2, **cfg)
    predictor.load_state_dict(torch.load(ck)); predictor.eval()
    i1, i2 = cfg['property_cols']
    for w in walks:
        novel, X, ei, ea = features(w, bool(cfg.get('ablate_bidir')))
        with torch.no_grad():
            Xp = X if cfg.get('feat_concat_W') else X[:, :mol_feats.shape[1]]
            u = do_predict(predictor, Xp, ei, ea).numpy() * std + mean
        rows.append((pd_, os.path.basename(ck), cfg['test_seed'], f'{names[i1]}/{names[i2]}', w, float(u[0]), float(u[1]), int(novel)))
with open(out_csv, 'w') as f:
    wr = csv.writer(f); wr.writerow(['predictor', 'ckpt', 'test_seed', 'pair', 'walk', 'perm', 'sel', 'novel'])
    wr.writerows(rows)
print('wrote', out_csv, len(rows), 'rows', flush=True)
