"""
Clean SMILES for the ancestral-sampling runs (`gen.py random`), which store the walk but not the motif-graph edges
`clean_smiles.py` needs. The edges are re-derived deterministically from the walk by the same `verify_walk` call
`gen.py record()` makes, so nothing is resampled and the structures are built by the dataset's own builder.

The sampler applies no novelty or deduplication filter — `utils/graph.py:is_novel` returns True unconditionally —
so the concatenated output of several seeds is an i.i.d. sample stream, which is what the unified comparison
protocol needs.

usage: python results_v2/clean_random.py <out.txt> <random_n*_s*.json> [...]
Output lines: '<clean_smiles> <0|1 sanitized> <walk>' in input order, streams concatenated in the order given.
"""
import os, sys, json
H = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(H, 'code'); GC = os.path.join(H, 'data', 'group-contrib')
os.environ['dataset'] = GC
os.environ['GC_DIR'] = GC
os.environ['WALKS_FILE'] = os.path.join(GC, 'polymer_walks_v2_preprocess.txt')
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, 'scripts'))
import argparse, networkx as nx                                                      # noqa: E402
from diffusion import *                                                              # noqa: E402,F401,F403
from rdkit import Chem, RDLogger                                                     # noqa: E402
RDLogger.DisableLog('rdApp.*')
import regen_smiles_diag as diag                                                     # noqa: E402

out_path, srcs = sys.argv[1], sys.argv[2:]
cfg = json.load(open(os.path.join(H, 'grammar', 'config.json')))
cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
args = argparse.Namespace(**cfg)
args.dags_file = f'{GC}/dags.pkl'; args.data_file = args.walks_file = f'{GC}/polymer_walks_v2_preprocess.txt'
args.motifs_folder = f'{GC}/all_groups/'; args.extra_label_path = f'{GC}/all_groups/all_extra.txt'
args.predefined_graph_file = f'{GC}/red_graph.edgelist'
diffusion_args = {k[len('diffusion_'):]: v for (k, v) in args.__dict__.items() if 'diffusion' in k}
graph_nx = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
mols = load_mols(args.motifs_folder)
red_grps = annotate_extra(mols, args.extra_label_path)
r_lookup = r_member_lookup(mols)
_, dags = load_dags(args)
graph = DiffusionGraph(dags, graph_nx, **diffusion_args)
G = graph.graph

lines, n_fail = [], 0
for src in srcs:
    for r in json.load(open(src)):
        walk = r['walk']
        try:
            _, conn = verify_walk(r_lookup, G, walk.split('->'), loop_back=True)
            g = nx.DiGraph()
            for a, b, e in conn:
                A, B = a.val.split(':')[0], b.val.split(':')[0]
                g.add_node(a.id, val=A); g.add_node(b.id, val=B)
                if b.id != 0:                        # loop-back into the root is not an assembly step
                    g.add_edge(a.id, b.id, e=int(e))
            diag.DAGS = {0: (g, None, None)}
            m, _ = diag.build(0, fix=True)
            smi, ok = diag.clean_smiles(diag.strip_unused_r(m))
            ok = ok and Chem.MolFromSmiles(smi) is not None
        except Exception:
            smi, ok = 'BUILD_FAILED', False
            n_fail += 1
        lines.append(f'{smi} {int(ok)} {walk}')
open(out_path, 'w').write('\n'.join(lines) + '\n')
print(f"{sum(l.split()[1] == '1' for l in lines)}/{len(lines)} sanitized ({n_fail} build failures)", flush=True)
