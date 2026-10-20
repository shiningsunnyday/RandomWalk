"""
Secondary headline metric S, for one campaign arm: the ensemble spread on the 1000 generated structures and the
SI Figs S2-S4 screening counts, plus the dataset-polymer predictions used for the error analysis.

Same mechanics as results_v2/predict.py + ensemble.py + plots.py:screening_figures, extended to
  * K ensemble members (one per split seed) instead of exactly 3
  * --log_target runs (predictions are de-normalized in log space and then raised to the power of ten)

usage: python screening.py <arm> [more arms ...]
  reads  campaign/outputs/selected_<arm>.csv   (written by summarize_campaign.py)
  writes campaign/outputs/screen_<arm>_members.csv
         campaign/outputs/screen_<arm>_ensemble.txt      generated structures: '<walk> <prop> <mean> +- <std>'
         campaign/outputs/screen_<arm>_dataset.txt       the 114 dataset polymers, same format
         campaign/outputs/screen_<arm>.json              S, the screening counts, and the residual summary
"""
import os, sys, json, glob, argparse, csv, re
from collections import defaultdict

H = os.path.dirname(os.path.abspath(__file__))
RV = os.path.dirname(H)                                  # results_v2/
GC = os.path.join(RV, 'data', 'group-contrib')
os.environ.setdefault('dataset', GC)
sys.path.insert(0, os.path.join(H, 'code'))
os.chdir(os.path.join(H, 'code'))
import numpy as np, torch, networkx as nx
from main import *                                        # noqa
from utils.graph import is_novel, dfs_traverse

GASES = ['H2', 'N2', 'O2', 'CH4', 'CO2']
# gas -> (pair, Robeson 2008 k, n, (perm std threshold, selectivity std threshold))  [plots.py GAS]
GAS = {'H2': ('H2/N2', 9.765e4, -1.4841, (500, 10)), 'O2': ('O2/N2', 1.396e6, -5.666, (100, 2)),
       'CO2': ('CO2/CH4', 5.369e6, -2.636, (1000, 5))}
PROPS = ['H2', 'H2/N2', 'O2', 'O2/N2', 'CO2', 'CO2/CH4']

args = argparse.Namespace(motifs_folder=GC + '/all_groups/', extra_label_path=GC + '/all_groups/all_extra.txt',
    grammar_folder=os.path.join(RV, 'grammar'), predefined_graph_file=GC + '/red_graph.edgelist',
    dags_file=GC + '/dags.pkl', walks_file=GC + '/polymer_walks_v2_preprocess.txt',
    mol_feat=['fp'], feat_concat_W=True, concat_mol_feats=False, cuda=-1, task='regression',
    ablate_bidir=False, attn_W=False, edge_weights=False)
data, dags = load_dags(args)
cfg_g = json.load(open(os.path.join(args.grammar_folder, 'config.json'))); cfg_g = json.loads(cfg_g) if isinstance(cfg_g, str) else cfg_g
diffusion_args = {k[len('diffusion_'):]: v for (k, v) in cfg_g.items() if 'diffusion' in k}
graph = DiffusionGraph(dags, nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph), **diffusion_args)
G = graph.graph
mols = load_mols(args.motifs_folder); annotate_extra(mols, args.extra_label_path)
feat_lookup = {name_group(i): mol2fp(mols[i - 1]) for i in range(1, len(mols) + 1)}
mol_feats = np.zeros((len(G), len(feat_lookup[list(feat_lookup)[0]])), dtype=np.float32)
for nd in G.nodes():
    mol_feats[graph.index_lookup[nd]] = feat_lookup[nd.split(':')[0]]
diffusion_args['adj_matrix'] = nx.adjacency_matrix(G).toarray()
gmodel = L_grammar(len(G), diffusion_args)
gmodel.load_state_dict(torch.load(os.path.join(args.grammar_folder, 'ckpt.pt'))); gmodel.eval()

gen_walks = [l.split()[0] for l in open(os.path.join(RV, 'outputs', 'screening_walks.txt')) if l.strip()]
ds_lines = [l for l in open(args.walks_file).read().split('\n') if l.strip()]
ds_walks = [re.sub(r'-\d+>', '->', l.split()[1]) for l in ds_lines]
ds_names = [l.split()[0] for l in ds_lines]
dag_of = {w: d for w, d in zip(ds_walks, sorted(dags, key=lambda d: d.dag_id))}


def split_walk(w):
    toks, buf, depth, i = [], '', 0, 0
    while i < len(w):
        if w.startswith('->', i) and depth == 0:
            toks.append(buf); buf = ''; i += 2; continue
        depth += (w[i] == '[') - (w[i] == ']'); buf += w[i]; i += 1
    return toks + [buf]


def tuplify(nd):
    nd.children = [c if isinstance(c, tuple) else (c, 0) for c in nd.children]
    if nd.parent is not None and not isinstance(nd.parent, tuple): nd.parent = (nd.parent, 0)
    for c in nd.children: tuplify(c[0])


cache = {}
def features(w, directed):
    k = (w, directed)
    if k in cache: return cache[k]
    a2 = argparse.Namespace(**{**vars(args), 'ablate_bidir': directed})
    if w in dag_of:
        root = dag_of[w]; proc = graph.lookup_process(root.dag_id)
        cache[k] = (0,) + tuple(featurize_walk(a2, graph, gmodel, root, proc, mol_feats, feat_lookup))
    else:
        root, conn = dfs_traverse(split_walk(w), loop_back=True)
        for a, b in conn: assert b.val in G[a.val], (w, a.val, b.val)
        tuplify(root); DiffusionGraph.value_count(root, {})
        novel = is_novel(deepcopy(dags), root)
        proc = DiffusionProcess(root, graph.index_lookup, **diffusion_args)
        cache[k] = (novel,) + tuple(featurize_walk(a2, graph, gmodel, root, proc, mol_feats, feat_lookup))
    return cache[k]


def predict(pdirs, walks, member_rows=None):
    """-> {(walk, prop): [value per member]}; if member_rows is a list, per-member rows are appended to it"""
    vals = defaultdict(list)
    for pd_ in pdirs:
        cfg = json.load(open(os.path.join(pd_, 'config.json'))); cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
        cks = glob.glob(os.path.join(pd_, 'predictor_ckpt_*.pt'))
        ck = min(cks, key=lambda p: float(os.path.basename(p)[15:-3]))      # lowest (smoothed) validation loss
        ms = np.atleast_2d(np.loadtxt(os.path.join(pd_, 'mean_and_std.txt'))); mean, std = ms[:, 0], ms[:, 1]
        logt = bool(cfg.get('log_target', False))
        predictor = Predictor(num_heads=2, **cfg)
        predictor.load_state_dict(torch.load(ck)); predictor.eval()
        i1, i2 = cfg['property_cols']
        names = (GASES[i1], f'{GASES[i1]}/{GASES[i2]}')
        for w in walks:
            _, X, ei, ea = features(w, bool(cfg.get('ablate_bidir')))
            with torch.no_grad():
                Xp = X if cfg.get('feat_concat_W') else X[:, :mol_feats.shape[1]]
                u = do_predict(predictor, Xp, ei, ea).numpy() * std + mean
            if logt: u = np.power(10.0, np.clip(u, -8.0, 9.0))
            for j, nm in enumerate(names):
                vals[(w, nm)].append(float(u[j]))
                if member_rows is not None:
                    member_rows.append(dict(dir=pd_, seed=cfg['test_seed'], walk=w, prop=nm, pred=float(u[j])))
    return vals


def ens(vals, walks):
    lines, stat = [], {}
    for w in walks:
        for p in PROPS:
            if (w, p) in vals:
                x = np.array(vals[(w, p)])
                lines.append(f'{w} {p} {x.mean()} +- {x.std()}')
                stat.setdefault(p, []).append((x.mean(), x.std(), len(x)))
    return lines, stat


def run(arm):
    sel = list(csv.DictReader(open(os.path.join(H, 'outputs', f'selected_{arm}.csv'))))
    pdirs = sorted({r['dir'] for r in sel})
    print(f'[{arm}] {len(pdirs)} selected member runs', flush=True)
    gv = predict(pdirs, gen_walks)
    dmem = []
    dv = predict(pdirs, ds_walks, member_rows=dmem)
    with open(os.path.join(H, 'outputs', f'screen_{arm}_dataset_members.csv'), 'w') as f:
        wr = csv.DictWriter(f, fieldnames=['dir', 'seed', 'walk', 'prop', 'pred']); wr.writeheader(); wr.writerows(dmem)
    gl, gs = ens(gv, gen_walks); dl, ds = ens(dv, ds_walks)
    open(os.path.join(H, 'outputs', f'screen_{arm}_ensemble.txt'), 'w').write('\n'.join(gl) + '\n')
    open(os.path.join(H, 'outputs', f'screen_{arm}_dataset.txt'), 'w').write('\n'.join(dl) + '\n')

    res = {'arm': arm, 'n_members_per_pair': len(pdirs) // 3, 'n_generated': len(gen_walks)}
    res['median_ensemble_std'] = {p: float(np.median([s for _, s, _ in gs[p]])) for p in PROPS}
    res['median_relative_std'] = {p: float(np.median([s / abs(m) if m else np.nan for m, s, _ in gs[p]])) for p in PROPS}
    res['frac_negative_mean'] = {p: float(np.mean([m <= 0 for m, _, _ in gs[p]])) for p in PROPS}
    # SI Figs S2-S4 counts, exactly plain plots.py:screening_figures logic
    counts = {}
    for gas, (pair, k, n, (tp, ts)) in GAS.items():
        P = np.array([[gs[gas][i][0], gs[gas][i][1]] for i in range(len(gen_walks))])
        S = np.array([[gs[pair][i][0], gs[pair][i][1]] for i in range(len(gen_walks))])
        keep = (P[:, 1] < tp) & (S[:, 1] < ts) & (P[:, 0] > 0) & (S[:, 0] > 0)
        counts[gas] = int(keep.sum())
    res['screening_counts'] = counts
    # dataset residuals (for the error analysis), on the raw barrer scale
    gt = {}
    for nm, w, l in zip(ds_names, ds_walks, ds_lines):
        v = l.split()[-1].strip('()').split(',')
        f = lambda x: float('nan') if x.strip() in ('-', '_', '') else float(x.strip())
        vv = [f(x) for x in v]
        gt[w] = {'H2': vv[0], 'O2': vv[2], 'CO2': vv[4],
                 'H2/N2': vv[0] / vv[1] if np.isfinite(vv[1]) else np.nan,
                 'O2/N2': vv[2] / vv[1] if np.isfinite(vv[1]) else np.nan,
                 'CO2/CH4': vv[4] / vv[3] if np.isfinite(vv[3]) else np.nan}
    rows = []
    for nm, w in zip(ds_names, ds_walks):
        for p in PROPS:
            if (w, p) not in dv: continue
            t, pr = gt[w][p], float(np.mean(dv[(w, p)]))
            if not np.isfinite(t) or t <= 0: continue
            rows.append(dict(name=nm, walk=w, prop=p, true=t, pred=pr,
                             std=float(np.std(dv[(w, p)])),
                             log_ratio=float(np.log10(pr / t)) if pr > 0 else float('nan')))
    with open(os.path.join(H, 'outputs', f'screen_{arm}_residuals.csv'), 'w') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    res['dataset_negative_predictions'] = int(sum(1 for r in rows if r['pred'] <= 0))
    res['dataset_median_abs_log_ratio'] = {p: float(np.nanmedian([abs(r['log_ratio']) for r in rows if r['prop'] == p]))
                                           for p in PROPS}
    json.dump(res, open(os.path.join(H, 'outputs', f'screen_{arm}.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    for a in sys.argv[1:]: run(a)
