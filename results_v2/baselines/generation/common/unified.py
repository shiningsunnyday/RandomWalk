"""
Unified generation protocol for the baseline comparison table (Table B).

Table 1 of the paper reports the *best-first enumeration*: a sequential, stateful search that returns the
grammar's highest-confidence walks in order. No baseline here has such a mode, so comparing against it measures
the search, not the generative model. This script measures the thing every method does have — i.i.d. sampling —
under one protocol for all of them, ours included:

  (i)  independent N      draw N samples, report valid / unique / novel / diversity / size on those N
  (ii) until N unique     keep drawing from the same stream until N distinct molecules have been seen, and report
                          how many samples that took. A method that collapses onto a few molecules never gets
                          there; the cap makes that visible instead of hiding it inside a "unique" percentage.

Both protocols are derived from ONE sample stream per method, so (i) is the stream's first N and (ii) is the
prefix of the same stream that first contains N distinct molecules. Nothing is resampled between them and the
two rows of a method are not independent draws.

A "sample" is one molecule the method emits, valid or not, which is the only unit all of these share. For us a
sample is a closed walk the sampler accepted for assembly, so our Valid is the fraction of closed walks that
assemble and sanitize — a stricter number than Table 1's staged Valid (SMILES), which conditions on assembly
having succeeded. The two tables are not meant to agree: Table 1 reports the enumeration, this one reports i.i.d.
sampling under a protocol the baselines can also follow.

usage:
  python common/unified.py stream <method> <n>       # draw a stream, write unified/<method>/stream.txt
  python common/unified.py metrics [--n 1000] [--cap 20000]   # -> unified/unified.jsonl, unified/table.md
"""
import os, sys, json, argparse, subprocess, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RV = os.path.expanduser('~/RandomWalk/results_v2')
U = os.path.join(HERE, 'unified')
PY = {'pw': os.path.expanduser('~/miniconda3/envs/polymer_walk/bin/python'),
      'tm': os.path.expanduser('~/miniconda3/envs/torch_molecule/bin/python'),
      'ri': os.path.expanduser('~/miniconda3/envs/reinvent4/bin/python')}

# Each entry says how to draw `n` samples into <out>/stream.txt. Models are loaded from the checkpoints the
# original 1000-sample runs trained, so the generative model is identical across the two tables.
METHODS = {
    'smiles_lstm': dict(
        label='SMILES LSTM',
        cmd=lambda n, out: [PY['pw'], 'common/smiles_lm.py', '--arch', 'lstm', '--out', 'runs/smiles_lstm',
                            '--sample_only', '--nsamples', str(n), '--samples_name', 'stream.txt', '--seed', '101'],
        src='runs/smiles_lstm/stream.txt'),
    'smiles_transformer': dict(
        label='SMILES transformer',
        cmd=lambda n, out: [PY['pw'], 'common/smiles_lm.py', '--arch', 'transformer', '--out',
                            'runs/smiles_transformer', '--sample_only', '--nsamples', str(n),
                            '--samples_name', 'stream.txt', '--seed', '101'],
        src='runs/smiles_transformer/stream.txt'),
    'smiles_lstm_fullfit': dict(
        label='SMILES LSTM (early-stopped)',
        cmd=lambda n, out: [PY['pw'], 'common/smiles_lm.py', '--arch', 'lstm', '--out', 'runs/smiles_lstm_fullfit',
                            '--sample_only', '--nsamples', str(n), '--samples_name', 'stream.txt', '--seed', '101'],
        src='runs/smiles_lstm_fullfit/stream.txt'),
    'smiles_transformer_fullfit': dict(
        label='SMILES transformer (early-stopped)',
        cmd=lambda n, out: [PY['pw'], 'common/smiles_lm.py', '--arch', 'transformer', '--out',
                            'runs/smiles_transformer_fullfit', '--sample_only', '--nsamples', str(n),
                            '--samples_name', 'stream.txt', '--seed', '101'],
        src='runs/smiles_transformer_fullfit/stream.txt'),
    'hiervae': dict(
        label='Hier-VAE',
        cmd=lambda n, out: [PY['pw'], os.path.join(HERE, 'common', 'hier_sample.py'), '--vocab', 'vocab.txt',
                            '--model', 'ckpt/model.4', '--nsample', str(n), '--out', 'stream.txt'],
        cwd='runs/hiervae', src='runs/hiervae/stream.txt'),
    'graph_ga_uncond': dict(
        label='Graph GA (unconditional)',
        cmd=lambda n, out: [PY['pw'], 'common/graph_ga.py', '--mode', 'uncond', '--out', 'runs/graph_ga_stream',
                            '--nsamples', str(n), '--seed', '101'],
        src='runs/graph_ga_stream/samples.txt'),
    'graph_ga_co2': dict(
        label='Graph GA (CO2-conditioned)',
        cmd=lambda n, out: [PY['pw'], 'common/graph_ga.py', '--mode', 'co2', '--out', 'runs/graph_ga_co2_stream',
                            '--nsamples', str(n), '--seed', '101'],
        # the CO2 run's samples.txt is the top-k by surrogate score, which is an optimiser's output and not a
        # sample stream; under this protocol its stream is the offspring log in creation order, like the
        # unconditional run's
        src='runs/graph_ga_co2_stream/offspring_log.tsv', col=1),
    'reinvent_tl': dict(
        label='REINVENT 4 (transfer only)',
        cmd=lambda n, out: ['bash', os.path.join(HERE, 'common', 'reinvent_stream.sh'), 'tl', str(n)],
        src='runs/reinvent/stream_tl.csv', csv=True),
    'reinvent_rl': dict(
        label='REINVENT 4 (transfer + RL)',
        cmd=lambda n, out: ['bash', os.path.join(HERE, 'common', 'reinvent_stream.sh'), 'rl', str(n)],
        src='runs/reinvent/stream_rl.csv', csv=True),
    'jtvae': dict(
        label='JT-VAE',
        cmd=lambda n, out: [PY['tm'], 'common/jtvae_run.py', '--out', 'runs/jtvae', '--sample_only',
                            '--nsamples', str(n), '--samples_name', 'stream.txt'],
        src='runs/jtvae/stream.txt'),
    'deg': dict(
        label='DEG (grammar induction only)',
        cmd=None,                 # the published DEG reward needs one Retro* call per molecule; not resampled
        src='unified/deg/stream.txt'),
    'ours': dict(
        label='Ours (motif-walk, i.i.d. sampling)',
        cmd=None,                 # built by results_v2/clean_random.py from gen.py random runs
        src='unified/ours/stream.txt', col=0),
}


def stream(method, n):
    m = METHODS[method]
    out = os.path.join(U, method)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    cmd = m['cmd'](n, out)
    assert cmd, f'{method} has no sampler here; its stream is built elsewhere'
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=os.path.join(HERE, m.get('cwd', '.')), check=True)
    src = os.path.join(HERE, m['src'])
    lines = [l.rstrip('\n') for l in open(src)]
    if m.get('csv'):
        lines = [l.split(',')[0] for l in lines[1:]]            # REINVENT writes 'SMILES,NLL'
    lines = [l.split()[m['col']] if m.get('col') is not None and l.split() else l.strip() for l in lines]
    open(os.path.join(out, 'stream.txt'), 'w').write('\n'.join(lines) + '\n')
    json.dump({'method': method, 'n_requested': n, 'n_drawn': len(lines), 'seconds': time.time() - t0},
              open(os.path.join(out, 'stream_info.json'), 'w'), indent=1)
    print(f'{method}: {len(lines)} samples in {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- metrics

def canon(s):
    from rdkit import Chem
    if not s or s == '<empty>':
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def train_set():
    data = os.path.join(RV, 'data', 'group-contrib', 'polymer_walks_v2_smiles.txt')
    return {canon(l.split()[0]) for l in open(data) if l.strip()}


def score(samples, train, n):
    """metrics over a list of raw samples, in stream order"""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    import numpy as np
    can = [canon(s) for s in samples]
    ok = [c for c in can if c]
    uniq = list(dict.fromkeys(ok))
    mols = [Chem.MolFromSmiles(c) for c in uniq]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048) for m in mols]
    sim = sum(sum(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])) for i in range(len(fps)))
    div = 1 - sim / (len(fps) * (len(fps) - 1) / 2) if len(fps) > 1 else 0.0
    return {'n_samples': len(samples), 'valid': len(ok) / len(samples) if samples else 0.0,
            'unique': len(uniq) / len(samples) if samples else 0.0, 'n_unique': len(uniq),
            'novel': sum(c not in train for c in ok) / len(ok) if ok else 0.0,
            'novel_of_unique': sum(c not in train for c in uniq) / len(uniq) if uniq else 0.0,
            'diversity': float(div),
            'median_heavy_atoms': float(np.median([m.GetNumHeavyAtoms() for m in mols])) if mols else None}


def until_unique(samples, train, n, cap):
    """prefix of the stream that first contains n distinct molecules"""
    seen, drawn = set(), 0
    for i, s in enumerate(samples[:cap], start=1):
        c = canon(s)
        if c:
            seen.add(c)
        drawn = i
        if len(seen) >= n:
            break
    r = score(samples[:drawn], train, n)
    r.update({'reached': len(seen) >= n, 'samples_needed': drawn if len(seen) >= n else None,
              'unique_at_cap': len(seen), 'cap': cap})
    return r


def metrics(n, cap):
    train = train_set()
    rows = []
    for name in sorted(os.listdir(U)) if os.path.isdir(U) else []:
        p = os.path.join(U, name, 'stream.txt')
        if not os.path.exists(p):
            continue
        samples = [l.strip() for l in open(p)]
        label = METHODS.get(name, {}).get('label', name)
        row = {'method': name, 'label': label, 'n_stream': len(samples),
               'independent': score(samples[:n], train, n),
               'until_unique': until_unique(samples, train, n, cap)}
        # Our stream holds only the candidates the sampler accepted, so its validity is the rate at which an
        # accepted structure rebuilds from its walk string, not the method's validity. The sampler's own counters
        # give the comparable number: of the closed walks it offered for assembly, this fraction sanitized.
        vj = os.path.join(U, name, 'validity.json')
        if os.path.exists(vj):
            v = json.load(open(vj))
            row['independent']['valid'] = v['valid_rate']
            row['independent']['valid_source'] = f"sampler counters, {v['num_valid']}/{v['num_tried']}"

        rows.append(row)
        u = row['until_unique']
        need = str(u['samples_needed']) if u['reached'] else f">{u['n_samples']} (only {u['unique_at_cap']} unique)"
        print(f"{label:34s} indep: valid {100 * row['independent']['valid']:3.0f}% "
              f"unique {100 * row['independent']['unique']:3.0f}%   until-{n}-unique: {need}", flush=True)
    with open(os.path.join(U, 'unified.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    # markdown rendering, ours first then the baselines in the order of the existing Table S14
    order = ['ours', 'smiles_lstm', 'smiles_lstm_fullfit', 'smiles_transformer', 'smiles_transformer_fullfit',
             'hiervae', 'jtvae', 'graph_ga_uncond', 'graph_ga_co2', 'reinvent_tl', 'reinvent_rl', 'deg']
    rows.sort(key=lambda r: order.index(r['method']) if r['method'] in order else 99)
    pct = lambda x: '—' if x is None else (f'{100 * x:.1f}%' if 0 < 100 * x < 1 else f'{100 * x:.0f}%')
    L = [f'### Protocol (i): {n} independent samples', '',
         '| Method | Valid | Unique | Novel | Diversity | Median heavy atoms |', '|---|---|---|---|---|---|']
    for r in rows:
        d = r['independent']
        ha = '—' if d['median_heavy_atoms'] is None else f"{d['median_heavy_atoms']:.0f}"
        L.append(f"| {r['label']} | {pct(d['valid'])} | {pct(d['unique'])} | {pct(d['novel'])} | "
                 f"{d['diversity']:.2f} | {ha} |")
    L += ['', f'### Protocol (ii): sample until {n} distinct molecules', '',
          '| Method | Samples drawn | Reached | Distinct found | Novel | Diversity |', '|---|---|---|---|---|---|']
    for r in rows:
        u = r['until_unique']
        L.append(f"| {r['label']} | {u['n_samples']:,} | {'yes' if u['reached'] else 'no'} | "
                 f"{u['n_unique']:,} | {pct(u['novel_of_unique'])} | {u['diversity']:.2f} |")
    open(os.path.join(U, 'table.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'wrote {U}/unified.jsonl and table.md ({len(rows)} methods)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['stream', 'metrics'])
    ap.add_argument('args', nargs='*')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--cap', type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(U, exist_ok=True)
    if a.action == 'stream':
        stream(a.args[0], int(a.args[1]))
    else:
        metrics(a.n, a.cap)
