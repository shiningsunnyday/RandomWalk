"""
Chemprop D-MPNN prediction baseline under the v2 protocol (see bl_data.py).

Chemprop gets exactly our splits as --separate_val_path / --separate_test_path, so it picks its own best epoch on the
same validation polymers we do and never sees the test split during training. A small architecture grid plus the
log-target option is selected on validation loss; predictions are then scored with our metrics on the raw targets
normalized by training-split statistics.

Run with the reinvent4 environment, which has chemprop 1.5.2:
  ~/miniconda3/envs/reinvent4/bin/python bl_chemprop.py [--epochs 60] [--gpu 0]
"""
import os, sys, json, time, shutil, argparse, subprocess
import numpy as np
import pandas as pd

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import bl_data as D

SCRATCH = os.environ.get('BL_SCRATCH', '/tmp/claude-1026/-home-msun415/e4c142f8-9c54-42f0-bb59-09e87958531b/scratchpad/chemprop')

BIN = os.path.join(os.path.dirname(os.path.realpath(sys.executable)))   # chemprop_train lives next to the env python

GRID = [
    {'hidden_size': 300, 'depth': 3, 'dropout': 0.0, 'ffn_num_layers': 2},
    {'hidden_size': 300, 'depth': 3, 'dropout': 0.2, 'ffn_num_layers': 2},
    {'hidden_size': 600, 'depth': 5, 'dropout': 0.0, 'ffn_num_layers': 3},
    {'hidden_size': 150, 'depth': 2, 'dropout': 0.0, 'ffn_num_layers': 1},
]


def write_csv(path, smiles, y):
    pd.DataFrame({'smiles': smiles, 't0': y[:, 0], 't1': y[:, 1]}).to_csv(path, index=False)


def one_run(wd, cols, smiles, y, tr, va, te, cfg, log_target, epochs, gpu):
    os.makedirs(wd, exist_ok=True)
    yf = np.log(np.clip(y, 1e-12, None)) if log_target else y
    for tag, idx in (('train', tr), ('val', va), ('test', te)):
        write_csv(os.path.join(wd, f'{tag}.csv'), [smiles[i] for i in idx], yf[idx])
    cmd = [os.path.join(BIN, 'chemprop_train'), '--data_path', os.path.join(wd, 'train.csv'),
           '--separate_val_path', os.path.join(wd, 'val.csv'),
           '--separate_test_path', os.path.join(wd, 'test.csv'),
           '--dataset_type', 'regression', '--save_dir', os.path.join(wd, 'model'),
           '--smiles_columns', 'smiles', '--target_columns', 't0', 't1',
           '--epochs', str(epochs), '--metric', 'mse', '--quiet', '--seed', '0', '--pytorch_seed', '0']
    cmd += ['--gpu', str(gpu)] if gpu is not None and gpu >= 0 else ['--no_cuda']
    for k, v in cfg.items():
        cmd += [f'--{k}', str(v)]
    r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True,
                       env={**os.environ, 'OMP_NUM_THREADS': '1'})
    if r.returncode != 0:
        raise RuntimeError('chemprop_train failed:\n' + r.stdout[-3000:] + r.stderr[-3000:])
    out = {}
    for tag, idx in (('val', va), ('test', te)):
        pp = os.path.join(wd, f'pred_{tag}.csv')
        pr = subprocess.run([os.path.join(BIN, 'chemprop_predict'), '--test_path', os.path.join(wd, f'{tag}.csv'),
                             '--checkpoint_dir', os.path.join(wd, 'model'), '--preds_path', pp,
                             '--smiles_columns', 'smiles'] +
                            (['--gpu', str(gpu)] if gpu is not None and gpu >= 0 else ['--no_cuda']),
                            cwd=wd, capture_output=True, text=True,
                            env={**os.environ, 'OMP_NUM_THREADS': '1'})
        if pr.returncode != 0:
            raise RuntimeError('chemprop_predict failed:\n' + pr.stdout[-2000:] + pr.stderr[-2000:])
        p = pd.read_csv(pp)[['t0', 't1']].to_numpy(dtype=float)
        out[tag] = np.exp(np.clip(p, -50, 50)) if log_target else p
    return out['val'], out['test']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(H, 'outputs', 'results.jsonl'))
    ap.add_argument('--raw_only', action='store_true',
                    help='drop the log-target option, so the row is like-for-like with our raw-target numbers')
    ap.add_argument('--seeds', default=None,
                    help='comma-separated split seeds, overriding the frozen (1,2,3); use with a separate --out '
                         '(see bl_classical.py --seeds and bl_k10.py)')
    a = ap.parse_args()
    if a.seeds:
        D.SEEDS = tuple(int(x) for x in a.seeds.split(','))
        print(f'split seeds: {D.SEEDS}', flush=True)
    logs = (False,) if a.raw_only else (False, True)
    label = 'Chemprop D-MPNN' + (' [raw target]' if a.raw_only else '')
    for pair in D.pairs():
        rows = D.masked(pair)
        smiles = [r['smiles'] for r in rows]
        y = D.targets(pair)
        per_seed = []
        for seed in D.SEEDS:
            tr, va, te = D.split(pair, seed)
            _, mu, sd = D.normalize(y, tr)
            zv, zt = (y[va] - mu) / sd, (y[te] - mu) / sd
            best = None
            for ci, cfg in enumerate(GRID):
                for lg in logs:
                    wd = os.path.join(SCRATCH, f'{pair}_s{seed}_c{ci}_l{int(lg)}')
                    shutil.rmtree(wd, ignore_errors=True)
                    try:
                        pv, pt = one_run(wd, None, smiles, y, tr, va, te, cfg, lg, a.epochs, a.gpu)
                    except Exception as e:
                        print(f'  skip {pair} s{seed} c{ci} log={lg}: {str(e)[:200]}', flush=True)
                        continue
                    vl = float((((pv - mu) / sd - zv) ** 2).mean())
                    if best is None or vl < best[0]:
                        best = (vl, (pt - mu) / sd, cfg, lg)
                    shutil.rmtree(wd, ignore_errors=True)
            if best is None:
                raise SystemExit(f'every chemprop config failed for {pair} seed {seed}')
            vl, pz, cfg, lg = best
            mae, r2 = D.metrics(zt, pz)
            per_seed.append({'seed': seed, 'val_mse': vl, 'mae': mae, 'r2': r2,
                             'cfg': {k: str(v) for k, v in cfg.items()}, 'log_target': lg})
            print(f'  {pair} s{seed} MAE {mae[0]:.3f}/{mae[1]:.3f} R2 {r2[0]:.3f}/{r2[1]:.3f} '
                  f'cfg={cfg} log={lg}', flush=True)
        row = {'baseline': label, 'feat': 'smiles (D-MPNN)', 'pair': pair, 'targets': list(D.TARGET_NAMES[pair]),
               'n': int(len(y)), 'per_seed': per_seed,
               'mae_mean': np.mean([s['mae'] for s in per_seed], axis=0).tolist(),
               'mae_std': np.std([s['mae'] for s in per_seed], axis=0).tolist(),
               'r2_mean': np.mean([s['r2'] for s in per_seed], axis=0).tolist(),
               'r2_std': np.std([s['r2'] for s in per_seed], axis=0).tolist(),
               'n_configs': len(logs) * len(GRID), 'epochs': a.epochs,
               'when': time.strftime('%Y-%m-%dT%H:%M:%S')}
        D.record(a.out, row)
        print(f"{label} {pair} MAE {row['mae_mean'][0]:.3f}±{row['mae_std'][0]:.3f} / "
              f"{row['mae_mean'][1]:.3f}±{row['mae_std'][1]:.3f}  R2 {row['r2_mean'][0]:.3f}±{row['r2_std'][0]:.3f} / "
              f"{row['r2_mean'][1]:.3f}±{row['r2_std'][1]:.3f}", flush=True)


if __name__ == '__main__':
    main()
