"""
Generation with the v2 grammar on the corrected dataset (results_v2 code/data/grammar).

  bestfirst N        best-first enumeration (diffusion.extract_all_walks): the N most likely walks that assemble into
                     a valid structure. Deterministic given the grammar. -> Table 1 (top 1000), Table S8, screening.
  random N SEED      ancestral sampling (diffusion.sample_walks, min_thresh 0.1) until N valid novel walks, seeded.
                     Its counters give Valid (Walk) / (SMARTS) / (SMILES) in Table 1.

Each kept walk is stored with the edge list verify_walk chose (node ids, groups, motif-graph edge id), so that
clean_smiles.py can build its structure with the same builder as the dataset SMILES.
usage: python results_v2/gen.py bestfirst 1000 | random 1000 0          -> results_v2/outputs/gen/
"""
import sys, os, json, pickle, argparse, time, random
H = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(H, 'code'); GC = os.path.join(H, 'data', 'group-contrib')
OUT = os.path.join(H, 'outputs', 'gen'); os.makedirs(OUT, exist_ok=True)
os.environ['dataset'] = GC
sys.path.insert(0, CODE); os.chdir(CODE)
from diffusion import *

mode = sys.argv[1]; n = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
cfg = json.load(open(os.path.join(H, 'grammar', 'config.json'))); cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
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


def record(named_traj, root, probs):
    """walk string + the edges verify_walk chose, re-derived deterministically from the named trajectory"""
    _, conn = verify_walk(r_lookup, G, named_traj, loop_back=True)
    edges = [(a.id, a.val.split(':')[0], b.id, b.val.split(':')[0], int(e)) for a, b, e in conn]
    return {'walk': '->'.join(named_traj), 'edges': edges, 'raw_smiles': root.smiles,
            'prob': float(np.prod(list(probs.values()))) if probs else None,
            'novel': bool(is_novel(deepcopy(seen_dags), root))}


t0 = time.time()
if mode == 'random':
    novel, new_dags, trajs, m = sample_walks(args, G, graph, seen_dags, model, all_nodes, r_lookup, diffusion_args,
                                             return_metrics=True, loop_back=True)
    print('GEN_METRICS', json.dumps(m), flush=True)
    json.dump(m, open(f'{OUT}/random_n{n}_s{seed}_counters.json', 'w'))
    json.dump([{'walk': x[0], 'raw_smiles': d.smiles} for x, d in zip(novel, new_dags)],
              open(f'{OUT}/random_n{n}_s{seed}.json', 'w'), indent=0)
elif mode == 'bestfirst':
    walks = extract_all_walks(args, G, graph, seen_dags, model, all_nodes, r_lookup, predefined_graph)
    recs = [record(process_good_traj(w[0], all_nodes), w[1], w[2]) for w in walks]
    json.dump(recs, open(f'{OUT}/bestfirst_n{n}.json', 'w'), indent=0)
    pickle.dump([(w[0], w[2]) for w in walks], open(f'{OUT}/bestfirst_n{n}_trajs.pkl', 'wb'))
print('time', time.time() - t0, flush=True)
