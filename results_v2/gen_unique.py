"""
Best-first generation that enumerates each closed cycle once -> outputs/gen/unique_n<N>.json

The plain `gen.py bestfirst N` enumeration treats a closed walk as its string, so the same cyclic repeat unit is
emitted once per starting motif and once per direction: `L22->S32->L22` and `S32->L22->S32` are the same ring read
from different places, and both are kept. That is what holds Table 1's Unique column down to 62% — of 1000 walks
only 751 are distinct cycles (see `uniqueness.py`), and the loss is bookkeeping rather than a generator repeating
itself. This script removes the bookkeeping: a closed walk is accepted only if its cycle, canonicalised over
rotation and reflection, has not been accepted before, and the search keeps going until N distinct cycles are found.

It does not claim to remove every duplicate structure. Two genuinely different cycles can still assemble into the
same molecule once unconsumed attachment sites are capped (126 of the 751 distinct cycles do), and those are left
in: the enumeration ranks walks, so collapsing distinct walks that happen to share a structure would mean dropping
a walk the grammar considers more probable than one that is kept.

Implementation note: rather than fork the 1800-line `code/diffusion.py`, this wraps `verify_walk`, which the
best-first loop calls on every candidate and whose failure path is already `continue`. A closed candidate whose
cycle has been seen raises `_Dup`, so the search skips it exactly as it skips a chemically invalid walk. The wrapper
is removed again before the accepted walks are recorded, so the output is built by the unmodified code path. Nothing
under `code/` is changed.

usage: python results_v2/gen_unique.py <N>            -> results_v2/outputs/gen/unique_n<N>.json
"""
import sys, os, json, pickle, argparse, time, random

H = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(H, 'code'); GC = os.path.join(H, 'data', 'group-contrib')
OUT = os.path.join(H, 'outputs', 'gen'); os.makedirs(OUT, exist_ok=True)
os.environ['dataset'] = GC
sys.path.insert(0, CODE); os.chdir(CODE)
import diffusion
from diffusion import *                                                                      # noqa: F401,F403

n = int(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

cfg = json.load(open(os.path.join(H, 'grammar', 'config.json')))
cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
args = argparse.Namespace(**cfg)
args.dags_file = f'{GC}/dags.pkl'; args.data_file = args.walks_file = f'{GC}/polymer_walks_v2_preprocess.txt'
args.motifs_folder = f'{GC}/all_groups/'; args.extra_label_path = f'{GC}/all_groups/all_extra.txt'
args.predefined_graph_file = f'{GC}/red_graph.edgelist'; args.log_folder = OUT + '/'
args.num_generate_samples = n; args.min_thresh = 0.1; args.max_thresh = 0.9; args.softmax = False
args.concat_mol_feats = False
diffusion_args = {k[len('diffusion_'):]: v for (k, v) in args.__dict__.items() if 'diffusion' in k}
graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
predefined_graph = nx.read_edgelist(args.predefined_graph_file, create_using=nx.MultiDiGraph)
mols = load_mols(args.motifs_folder)
red_grps = annotate_extra(mols, args.extra_label_path)
r_lookup = r_member_lookup(mols)
_, dags = load_dags(args)
graph = DiffusionGraph(dags, graph, **diffusion_args)
G = graph.graph; all_nodes = list(G.nodes())
diffusion_args['adj_matrix'] = nx.adjacency_matrix(G).toarray()
data, dags = load_dags(args); seen_dags = deepcopy(dags)
print('N nodes', len(G), flush=True)
model = L_grammar(len(G), diffusion_args)
model.load_state_dict(torch.load(os.path.join(H, 'grammar', 'ckpt.pt')))
graph.reset()


def canon_cycle(walk):
    """the closed walk as a cycle of motifs, canonical over rotation and reflection (as uniqueness.py does)"""
    s = list(walk)
    if len(s) < 2 or s[0] != s[-1]:
        return None
    s = s[:-1]
    rots = [tuple(s[i:] + s[:i]) for i in range(len(s))]
    return min(rots + [tuple(reversed(r)) for r in rots])


class _Dup(Exception):
    """raised for a closed walk whose cycle has already been accepted; the search treats it as an invalid walk"""


_orig_verify = diffusion.verify_walk
_seen_cycles = set()
_stats = {'closed': 0, 'dup': 0, 'invalid': 0}


def verify_walk_unique(r_lookup, G_, walk, loop_back=False):
    if not loop_back:
        return _orig_verify(r_lookup, G_, walk, loop_back=loop_back)
    key = canon_cycle(walk)
    _stats['closed'] += 1
    if key is not None and key in _seen_cycles:
        _stats['dup'] += 1
        raise _Dup(str(key))
    try:
        out = _orig_verify(r_lookup, G_, walk, loop_back=loop_back)
    except Exception:
        _stats['invalid'] += 1
        raise
    if key is not None:
        _seen_cycles.add(key)                 # only once it is known to assemble, so an invalid rotation of a
    return out                                # cycle does not block a valid one


t0 = time.time()
diffusion.verify_walk = verify_walk_unique
try:
    walks = extract_all_walks(args, G, graph, seen_dags, model, all_nodes, r_lookup, predefined_graph)
finally:
    diffusion.verify_walk = _orig_verify      # the records below are built by the unmodified path


def record(named_traj, root, probs):
    _, conn = verify_walk(r_lookup, G, named_traj, loop_back=True)
    edges = [(a.id, a.val.split(':')[0], b.id, b.val.split(':')[0], int(e)) for a, b, e in conn]
    return {'walk': '->'.join(named_traj), 'edges': edges, 'raw_smiles': root.smiles,
            'prob': float(np.prod(list(probs.values()))) if probs else None,
            'novel': bool(is_novel(deepcopy(seen_dags), root))}


recs = [record(process_good_traj(w[0], all_nodes), w[1], w[2]) for w in walks]
cyc = {canon_cycle(r['walk'].split('->')) for r in recs}
assert len(cyc) == len(recs), f'{len(recs)} walks but {len(cyc)} distinct cycles: dedup did not hold'
json.dump(recs, open(f'{OUT}/unique_n{n}.json', 'w'), indent=0)
pickle.dump([(w[0], w[2]) for w in walks], open(f'{OUT}/unique_n{n}_trajs.pkl', 'wb'))
json.dump(_stats, open(f'{OUT}/unique_n{n}_stats.json', 'w'), indent=1)
print(f'{len(recs)} walks, all distinct cycles; closures examined {_stats["closed"]}, '
      f'skipped as duplicate cycles {_stats["dup"]}, invalid {_stats["invalid"]}', flush=True)
print('time', time.time() - t0, flush=True)
