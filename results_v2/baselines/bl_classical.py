"""
Classical / feature-based prediction baselines under the v2 protocol (see bl_data.py).

Every baseline gets the same deal our model gets: it is fitted on the training split only, its hyperparameters are
chosen by validation loss alone, and the test split is only ever evaluated. `log_target` is part of every grid, so a
baseline that benefits from predicting log-permeability is allowed to; the chosen value is recorded. Metrics are
always computed on the raw targets normalized by training-split statistics, so they are directly comparable to the
heldout_* columns of our predictor runs.

usage: python bl_classical.py [--only NAME] [--out results.jsonl]
"""
import os, sys, json, time, argparse, warnings
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import bl_data as D

warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, ConstantKernel, WhiteKernel, RBF
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


class Tanimoto(Kernel):
    """k(x, y) = <x,y> / (|x|^2 + |y|^2 - <x,y>), the standard kernel for fingerprint GPs on small datasets."""

    def __init__(self, variance=1.0, variance_bounds=(1e-3, 1e3)):
        self.variance = variance
        self.variance_bounds = variance_bounds

    @property
    def hyperparameter_variance(self):
        return Hyperparameter('variance', 'numeric', self.variance_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        Y_ = X if Y is None else Y
        xy = X @ Y_.T
        xx = (X ** 2).sum(axis=1)[:, None]
        yy = (Y_ ** 2).sum(axis=1)[None, :]
        K = self.variance * xy / np.maximum(xx + yy - xy, 1e-12)
        if eval_gradient:
            if Y is not None:
                raise ValueError('gradient only for Y=None')
            return K, (K / self.variance)[:, :, None]
        return K

    def diag(self, X):
        return np.full(X.shape[0], self.variance)

    def is_stationary(self):
        return False


def _grid(d):
    """Cartesian product of a dict of lists, as a list of dicts."""
    import itertools
    keys = list(d)
    return [dict(zip(keys, v)) for v in itertools.product(*[d[k] for k in keys])]


# ---- (feature kind, estimator factory, hyperparameter grid) per baseline
def _xgb(**kw):
    return MultiOutputRegressor(XGBRegressor(n_jobs=1, random_state=0, verbosity=0, **kw))


BASELINES = {
    # R1 asked for a heuristic polymer-design comparison: classical group contribution is a linear model over the
    # group counts of the repeat unit, which is exactly a ridge fit on the expert motif-count vector.
    'GC regression (motif counts)': ('motif_expert',
        lambda alpha, fit_intercept, **_: MultiOutputRegressor(Ridge(alpha=alpha, fit_intercept=fit_intercept)),
        {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0], 'fit_intercept': [True, False]}),
    # the ICML 2024 Bag-of-Motifs baseline: XGBoost on motif counts. '+expert' uses the 99 curated motifs, the
    # non-expert version an automatically extracted (BRICS) motif vocabulary.
    'Bag-of-Motifs': ('motif_brics',
        lambda n_estimators, max_depth, learning_rate, **_: _xgb(n_estimators=n_estimators, max_depth=max_depth,
                                                                 learning_rate=learning_rate),
        {'n_estimators': [16, 100, 400], 'max_depth': [3, 6, 10], 'learning_rate': [0.05, 0.3, 1.0]}),
    'Bag-of-Motifs (+expert)': ('motif_expert',
        lambda n_estimators, max_depth, learning_rate, **_: _xgb(n_estimators=n_estimators, max_depth=max_depth,
                                                                 learning_rate=learning_rate),
        {'n_estimators': [16, 100, 400], 'max_depth': [3, 6, 10], 'learning_rate': [0.05, 0.3, 1.0]}),
    'Morgan FP + random forest': ('morgan',
        lambda n_estimators, max_features, min_samples_leaf, **_: RandomForestRegressor(
            n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_samples_leaf,
            random_state=0, n_jobs=2),
        {'n_estimators': [500], 'max_features': ['sqrt', 0.1, 0.3], 'min_samples_leaf': [1, 2, 5]}),
    'Morgan FP + GP (Tanimoto)': ('morgan',
        lambda noise, **_: MultiOutputRegressor(GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Tanimoto() + WhiteKernel(noise, (1e-5, 1e1)),
            normalize_y=False, random_state=0, n_restarts_optimizer=2)),
        {'noise': [1e-3, 1e-2, 1e-1]}),
    'Morgan FP + SVR (RBF)': ('morgan',
        lambda C, gamma, epsilon, **_: MultiOutputRegressor(SVR(C=C, gamma=gamma, epsilon=epsilon)),
        {'C': [1.0, 10.0, 100.0], 'gamma': ['scale', 1e-3], 'epsilon': [0.01, 0.1]}),
    'RDKit descriptors + XGBoost': ('rdkit_desc',
        lambda n_estimators, max_depth, learning_rate, **_: _xgb(n_estimators=n_estimators, max_depth=max_depth,
                                                                 learning_rate=learning_rate),
        {'n_estimators': [100, 400], 'max_depth': [3, 6], 'learning_rate': [0.05, 0.1, 0.3]}),
    'RDKit descriptors + ridge': ('rdkit_desc',
        lambda alpha, **_: make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=alpha))),
        {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}),
    # polymer-pretrained and chemistry-pretrained language model embeddings + a linear head (the standard way these
    # checkpoints are used for small-data property prediction)
    'polyBERT + ridge': ('polybert',
        lambda alpha, **_: make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=alpha))),
        {'alpha': [1.0, 10.0, 100.0, 1000.0, 1e4]}),
    'polyBERT + GP (RBF)': ('polybert',
        lambda noise, **_: make_pipeline(StandardScaler(), MultiOutputRegressor(GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(10.0, (1e-2, 1e4)) + WhiteKernel(noise, (1e-5, 1e1)),
            normalize_y=False, random_state=0, n_restarts_optimizer=2))),
        {'noise': [1e-2, 1e-1]}),
    'ChemBERTa + ridge': ('chemberta',
        lambda alpha, **_: make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=alpha))),
        {'alpha': [1.0, 10.0, 100.0, 1000.0, 1e4]}),
    # R^2 = 0 reference: always predict the training mean. Anything at or below this has learned nothing.
    'Training mean (reference)': ('motif_expert',
        lambda **_: MultiOutputRegressor(Ridge(alpha=1e12, fit_intercept=True)),
        {'dummy': [0]}),
}


def fit_eval(X, y, tr, va, te, make, cfg, log_target):
    """Fit on train, return (val_mse, test_pred, val_pred) all on the normalized raw scale."""
    _, mu_raw, sd_raw = D.normalize(y, tr)
    yf = np.log(np.clip(y, 1e-12, None)) if log_target else y
    mu, sd = yf[tr].mean(axis=0), yf[tr].std(axis=0)
    z = (yf - mu) / sd
    est = make(**cfg)
    est.fit(X[tr], z[tr])

    def back(idx):
        p = np.asarray(est.predict(X[idx])).reshape(len(idx), y.shape[1]) * sd + mu
        if log_target:
            p = np.exp(np.clip(p, -50, 50))
        return (p - mu_raw) / sd_raw

    zt_va, zt_te = (y[va] - mu_raw) / sd_raw, (y[te] - mu_raw) / sd_raw
    p_va, p_te = back(va), back(te)
    return float(((p_va - zt_va) ** 2).mean()), p_te, zt_te


def run_one(label, out_path, logs=(False, True), suffix=''):
    kind, make, grid = BASELINES[label]
    cfgs = [(c, lg) for c in _grid(grid) for lg in logs]
    for pair in D.pairs():
        X = D.feat(kind, pair)
        y = D.targets(pair)
        per_seed = []
        for seed in D.SEEDS:
            tr, va, te = D.split(pair, seed)
            best = None
            for cfg, lg in cfgs:
                try:
                    vl, p_te, y_te = fit_eval(X, y, tr, va, te, make, cfg, lg)
                except Exception as e:                       # a config that cannot be fitted is simply not selected
                    continue
                if best is None or vl < best[0]:
                    best = (vl, p_te, y_te, cfg, lg)
            vl, p_te, y_te, cfg, lg = best
            mae, r2 = D.metrics(y_te, p_te)
            per_seed.append({'seed': seed, 'val_mse': vl, 'mae': mae, 'r2': r2,
                             'cfg': {k: str(v) for k, v in cfg.items()}, 'log_target': lg})
        row = {'baseline': label + suffix, 'feat': kind, 'pair': pair, 'targets': list(D.TARGET_NAMES[pair]),
               'n': int(len(y)), 'per_seed': per_seed,
               'mae_mean': np.mean([s['mae'] for s in per_seed], axis=0).tolist(),
               'mae_std': np.std([s['mae'] for s in per_seed], axis=0).tolist(),
               'r2_mean': np.mean([s['r2'] for s in per_seed], axis=0).tolist(),
               'r2_std': np.std([s['r2'] for s in per_seed], axis=0).tolist(),
               'n_configs': len(cfgs), 'when': time.strftime('%Y-%m-%dT%H:%M:%S')}
        D.record(out_path, row)
        print(f"{label + suffix:34s} {pair:4s} MAE {row['mae_mean'][0]:.3f}±{row['mae_std'][0]:.3f} / "
              f"{row['mae_mean'][1]:.3f}±{row['mae_std'][1]:.3f}  "
              f"R2 {row['r2_mean'][0]:.3f}±{row['r2_std'][0]:.3f} / {row['r2_mean'][1]:.3f}±{row['r2_std'][1]:.3f}",
              flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--out', default=os.path.join(H, 'outputs', 'results.jsonl'))
    ap.add_argument('--raw_only', action='store_true',
                    help='restrict every grid to the untransformed target, as our own model is; labels get " [raw target]"')
    ap.add_argument('--seeds', default=None,
                    help='comma-separated split seeds, overriding the frozen (1,2,3). Use with a separate --out: the '
                         'campaign showed three seeds is too few to estimate R2, and "1,2,3,4,5,6,7,8,9,10" is the '
                         'seed set its K=10 arms used, so the two are directly comparable.')
    a = ap.parse_args()
    if a.seeds:
        D.SEEDS = tuple(int(x) for x in a.seeds.split(','))
        print(f'split seeds: {D.SEEDS}', flush=True)
    logs, suffix = ((False,), ' [raw target]') if a.raw_only else ((False, True), '')
    for label in ([a.only] if a.only else list(BASELINES)):
        t0 = time.time()
        run_one(label, a.out, logs, suffix)
        print(f'  -> {label}{suffix} done in {time.time() - t0:.0f}s', flush=True)
