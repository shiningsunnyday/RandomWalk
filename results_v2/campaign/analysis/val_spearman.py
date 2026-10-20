"""
Validation Spearman rho (and validation raw-space R^2) for runs whose metrics.csv predates the campaign's
metrics patch — recovered exactly, without retraining, by loading each run's saved checkpoint and predicting
its own validation polymers.

Why: PLAN.md's original, pre-committed arm-selection rule was mean VALIDATION Spearman rho, which is
invariant to the target transform and therefore the only rule that compares a raw-target arm with a
log-target arm fairly. It was amended on 2026-10-19 21:15 to mean validation raw-space R^2 only because the
two control arms were launched minutes before the metrics patch and had no Spearman columns. That amendment
was a mistake of substance, not just of convenience: raw-barrer-space R^2 IS the training objective of a
raw-target run (up to the affine normalization) and is not the training objective of a log-target run, so the
rule systematically favours the raw-target arms. This script removes the need for the amendment by recovering
the missing column from the checkpoints, so the ORIGINAL rule can be applied to every arm.

Each run's checkpoint is the one its own validation loss selected (predictor_ckpt_<loss>.pt, lowest), which is
exactly the checkpoint the protocol uses; val_idx.txt is correct in every run dir (only test_idx.txt is
corrupted in the frozen dirs, and this script never reads it).

usage: python analysis/val_spearman.py <arm> [<arm> ...]
  -> campaign/outputs/val_spearman.csv  (appended/updated; columns dir,arm,pair,seed,prop,rho,r2_raw,r2_norm)
"""
import os, sys, json, glob, csv, re, argparse
A = os.path.dirname(os.path.abspath(__file__))
H = os.path.dirname(A)
RV = os.path.dirname(H)
GC = os.path.join(RV, 'data', 'group-contrib')
os.environ.setdefault('dataset', GC)
sys.path.insert(0, os.path.join(H, 'code')); sys.path.insert(0, A)
os.chdir(os.path.join(H, 'code'))
import numpy as np, torch, networkx as nx
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from main import *                                        # noqa
from splits import subset_lines

GASES = ['H2', 'N2', 'O2', 'CH4', 'CO2']
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

lines = [l for l in open(args.walks_file).read().split('\n') if l.strip()]
f1 = lambda x: float('nan') if x.strip() in ('-', '_', '') else float(x.strip())
props = np.array([[f1(x) for x in l.split()[-1].strip('()').split(',')] for l in lines])
dag_by_line = {d.dag_id: d for d in dags}

_fc = {}
def features(line_i, directed):
    k = (line_i, directed)
    if k not in _fc:
        root = dag_by_line[line_i]; proc = graph.lookup_process(root.dag_id)
        a2 = argparse.Namespace(**{**vars(args), 'ablate_bidir': directed})
        _fc[k] = tuple(featurize_walk(a2, graph, gmodel, root, proc, mol_feats, feat_lookup))
    return _fc[k]


rows = []
for arm in sys.argv[1:]:
    dirs = sorted(glob.glob(os.path.join(H, 'outputs', arm, '*', 'predictor_*', 'config.json')))
    print(f'[{arm}] {len(dirs)} runs', flush=True)
    for cf in dirs:
        d = os.path.dirname(cf)
        cfg = json.load(open(cf)); cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
        cks = glob.glob(os.path.join(d, 'predictor_ckpt_*.pt'))
        if not cks or not os.path.exists(os.path.join(d, 'val_idx.txt')): continue
        ck = min(cks, key=lambda p: float(os.path.basename(p)[15:-3]))
        ms = np.atleast_2d(np.loadtxt(os.path.join(d, 'mean_and_std.txt'))); mean, std = ms[:, 0], ms[:, 1]
        logt = bool(cfg.get('log_target', False))
        i1, i2 = cfg['property_cols']
        sl = subset_lines(props, i1, i2)
        vi = [int(x) for x in open(os.path.join(d, 'val_idx.txt'))]
        pred = Predictor(num_heads=2, **cfg); pred.load_state_dict(torch.load(ck)); pred.eval()
        yh, yt = [], []
        for p in vi:
            li = sl[p]
            X, ei, ea = features(li, bool(cfg.get('ablate_bidir')))
            with torch.no_grad():
                Xp = X if cfg.get('feat_concat_W') else X[:, :mol_feats.shape[1]]
                u = do_predict(pred, Xp, ei, ea).numpy() * std + mean
            if logt: u = np.power(10.0, np.clip(u, -8.0, 9.0))
            yh.append(u)
            yt.append([props[li, i1], props[li, i1] / props[li, i2]])
        yh, yt = np.array(yh), np.array(yt)
        names = (GASES[i1], f'{GASES[i1]}/{GASES[i2]}')
        for j, nm in enumerate(names):
            ok = np.isfinite(yh[:, j]) & np.isfinite(yt[:, j])
            rows.append(dict(dir=d, arm=arm, pair=f'{GASES[i1]}_{GASES[i2]}', seed=cfg['test_seed'], prop=nm,
                             fw=bool(cfg['feat_concat_W']),
                             rho=float(spearmanr(yt[ok, j], yh[ok, j]).correlation),
                             r2_raw=float(r2_score(yt[ok, j], yh[ok, j])) if ok.sum() > 2 else float('nan'),
                             n=int(ok.sum())))
    print(f'[{arm}] done, {len(rows)} rows so far', flush=True)

out = os.path.join(H, 'outputs', 'val_spearman.csv')
old = {}
if os.path.exists(out):
    for r in csv.DictReader(open(out)): old[(r['dir'], r['prop'])] = r
for r in rows: old[(r['dir'], r['prop'])] = r
with open(out, 'w') as f:
    w = csv.DictWriter(f, fieldnames=['dir', 'arm', 'pair', 'seed', 'prop', 'fw', 'rho', 'r2_raw', 'n'])
    w.writeheader(); w.writerows(old.values())
print('wrote', out, len(old), 'rows')
