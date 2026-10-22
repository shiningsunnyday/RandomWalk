"""
Constrained enumeration of repeat units: every walk it emits closes by construction.

The sampler this replaces proposed trajectories and discarded the ones that failed to return to their starting
motif, which is where almost all of its work went. Closure is a reachability property of the motif graph, so it can
be decided before a step is taken rather than after a trajectory is finished:

  * A bond consumes one R group at each end, so a motif in the interior of a walk needs at least two R groups and a
    motif can never leave through the R group it arrived on. The state of a partial walk is therefore the pair
    (motif, R group consumed on arrival), and the legal moves from it are the motif-graph edges leaving the motif
    through any *other* R group.
  * On that state graph, `dist[(X, p)][(v1, z)]` is the least number of further bonds needed to get from (X, p)
    back to the starting motif arriving at R group z. It is precomputed once by breadth-first search.
  * A partial walk of length k towards a target length T is extended only through moves whose resulting state can
    still reach an open R group of the starting motif within the remaining T − k − 1 steps.

Nothing that enters the queue can fail to close, so the generator ships only valid repeat units and validity is a
property of the construction rather than a rate to be reported. Walks are enumerated for T = 2, 3, 4 and carry the
motif-graph edge chosen at each step, in the dataset's own notation (`L3-2>S32`), so a walk determines its
structure rather than leaving the attachment to be re-derived.

usage: python results_v2/gen_constrained.py <T_max> [--cap N] [--out outputs/gen/constrained.json]
"""
import os, sys, json, time, argparse, collections
from heapq import heappush, heappop

H = os.path.dirname(os.path.abspath(__file__))
GC = os.path.join(H, 'data', 'group-contrib')
os.environ.setdefault('dataset', GC)
sys.path.insert(0, os.path.join(H, 'code'))
import networkx as nx                                                                # noqa: E402
from utils.preprocess import load_mols, annotate_extra                               # noqa: E402
from utils.graph import list_red_members                                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('T_max', type=int, nargs='?', default=4)
ap.add_argument('--cap', type=int, default=200000, help='stop after this many distinct cycles')
ap.add_argument('--out', default=os.path.join(H, 'outputs', 'gen', 'constrained.json'))
cli = ap.parse_args()


def load():
    G = nx.read_edgelist(os.path.join(GC, 'red_graph.edgelist'), create_using=nx.MultiDiGraph)
    mols = load_mols(os.path.join(GC, 'all_groups') + '/')
    annotate_extra(mols, os.path.join(GC, 'all_groups', 'all_extra.txt'))
    names = [l.strip() for l in open(os.path.join(GC, 'all_groups', 'names.txt')) if l.strip()]
    idx = {n: i for i, n in enumerate(names)}
    ports = {n: [tuple(sorted(g)) for g in list_red_members(mols[idx[n]])] for n in names if n in G}
    return G, ports


def port_graph(G, ports):
    """(motif, arrival port) -> [(motif, arrival port, edge key, departure port)], and the same keyed by departure"""
    out = collections.defaultdict(list)
    by_departure = collections.defaultdict(list)
    for u, v, k, d in G.edges(keys=True, data=True):
        if u not in ports or v not in ports:
            continue
        r1, r2 = tuple(sorted(d['r_grp_1'])), tuple(sorted(d['r_grp_2']))
        if r1 not in ports[u] or r2 not in ports[v]:
            continue
        pu, pv = ports[u].index(r1), ports[v].index(r2)
        by_departure[(u, pu)].append((v, pv, k))
        for p_in in range(len(ports[u])):
            if p_in != pu:                       # a walk cannot leave through the R group it arrived on
                out[(u, p_in)].append((v, pv, k, pu))
    return out, by_departure


def distances(moves, states):
    """dist[target][state] = fewest bonds from state to target, by breadth-first search on the reversed moves"""
    rev = collections.defaultdict(list)
    for s, outs in moves.items():
        for (v, pv, k, pu) in outs:
            rev[(v, pv)].append(s)
    dist = {}
    for tgt in states:
        d = {tgt: 0}
        q = collections.deque([tgt])
        while q:
            cur = q.popleft()
            for prev in rev.get(cur, ()):  # noqa: E501
                if prev not in d:
                    d[prev] = d[cur] + 1
                    q.append(prev)
        dist[tgt] = d
    return dist


def canon(cycle):
    """a cycle of (motif, edge key) steps, canonical over rotation and reflection"""
    n = len(cycle)
    rots = [tuple(cycle[i:] + cycle[:i]) for i in range(n)]
    # a reflection reverses the order of the motifs and of the bonds between them
    rev = [(cycle[(i - 1) % n][0], cycle[(i - 1) % n][1]) for i in range(n)][::-1]
    rots += [tuple(rev[i:] + rev[:i]) for i in range(n)]
    return min(rots)


def enumerate_cycles(T, moves, ports, dist, seen, out, t0):
    """every closed walk of exactly T bonds, each one closing by construction"""
    starts = [n for n, p in ports.items() if len(p) >= 2]
    for v1 in starts:
        targets = [(v1, z) for z in range(len(ports[v1]))]
        for a in range(len(ports[v1])):                      # the R group the walk leaves v1 through
            open_targets = [tg for tg in targets if tg[1] != a]
            stack = [((v1, a), 0, [])]
            while stack:
                (cur, p_in), k, path = stack.pop()
                if len(out) >= cli.cap:
                    return
                rem = T - k
                for (v, pv, key, pu) in moves.get((cur, p_in), ()) if k else \
                        [(v, pv, key, a) for (v, pv, key) in BY_DEP.get((v1, a), ())]:
                    if k and pu == p_in:
                        continue
                    step = path + [(cur, key)]
                    if k + 1 == T:
                        if v == v1 and pv != a:              # closed, and not back through the departure R group
                            cyc = canon(step)
                            if cyc not in seen:
                                seen.add(cyc)
                                out.append({'T': T, 'cycle': [[m, int(e)] for m, e in step], 'start': v1})
                        continue
                    # lookahead: can (v, pv) still reach an open R group of v1 in the steps that remain?
                    if any(dist[tg].get((v, pv), 1 << 30) <= rem - 1 for tg in open_targets):
                        stack.append(((v, pv), k + 1, step))


def main():
    t0 = time.time()
    G, ports = load()
    moves, by_dep = port_graph(G, ports)
    global BY_DEP
    BY_DEP = by_dep
    states = [(n, p) for n, ps in ports.items() for p in range(len(ps))]
    print(f'{len(ports)} motifs, {sum(len(v) for v in moves.values())} port transitions, '
          f'{len(states)} states ({time.time() - t0:.0f}s)', flush=True)
    dist = distances(moves, states)
    print(f'distances precomputed ({time.time() - t0:.0f}s)', flush=True)
    seen, out = set(), []
    for T in range(2, cli.T_max + 1):
        before = len(out)
        enumerate_cycles(T, moves, ports, dist, seen, out, t0)
        print(f'T = {T}: {len(out) - before} new cycles, {len(out)} total ({time.time() - t0:.0f}s)', flush=True)
        if len(out) >= cli.cap:
            print('cap reached', flush=True)
            break
    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    json.dump(out, open(cli.out, 'w'), indent=0)
    print(f'wrote {cli.out}: {len(out)} cycles', flush=True)


if __name__ == '__main__':
    main()
