"""
Apply the campaign's backward-compatible additions to campaign/code/main.py (a copy of the frozen
results_v2/code/main.py; the shared repo code and the frozen snapshot are untouched).

Every addition defaults to the frozen behaviour:
  --log_target        off      regress log10 of both targets
  --strata_file PATH  none     integer stratum label per walks_file line (114 lines)
  --stratify MODE     none     none | balanced (proportional stratified train/val/test) | holdout (one stratum = test)
  --holdout_stratum K  0       which stratum is the test split when --stratify holdout
  --huber DELTA       0        >0: HuberLoss(delta) instead of MSELoss
  --weight_decay W    0        Adam weight decay
  --val_smooth K      1        checkpoint/epoch selection on the trailing K-epoch mean of the validation loss
plus new metrics.csv columns (never replacing existing ones):
  val_loss_smooth, {,heldout_}<prop>_spearman, {,heldout_}raw_<prop>_{r^2,mae,mse}

usage: python patch_main.py            (idempotent; refuses to run twice)
"""
import os, sys, re

H = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(H, 'code', 'main.py')
src = open(P).read()
if 'CAMPAIGN PATCH' in src:
    print('already patched'); sys.exit(0)


def sub1(old, new):
    global src
    assert src.count(old) == 1, (src.count(old), old[:90])
    src = src.replace(old, new)


# ---------------------------------------------------------------- 1. argparse
sub1("""    parser.add_argument('--train_norm', action='store_true', help='normalize properties with training-split statistics instead of all labelled data')
""",
     """    parser.add_argument('--train_norm', action='store_true', help='normalize properties with training-split statistics instead of all labelled data')
    # ---- CAMPAIGN PATCH: all default to the frozen behaviour
    parser.add_argument('--log_target', action='store_true', help='regress log10 of the targets (permeability and selectivity) instead of raw barrer / raw ratio')
    parser.add_argument('--strata_file', default=None, help='file with one integer stratum label per walks_file line, for --stratify')
    parser.add_argument('--stratify', default='none', choices=['none', 'balanced', 'holdout'], help='split construction: plain seeded shuffle (none), proportional stratified train/val/test (balanced), or one whole stratum as the test split (holdout)')
    parser.add_argument('--holdout_stratum', default=0, type=int, help='which stratum is the test split when --stratify holdout')
    parser.add_argument('--huber', default=0.0, type=float, help='>0: HuberLoss(delta=this) as the training loss instead of MSELoss')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Adam weight decay')
    parser.add_argument('--val_smooth', default=1, type=int, help='select the epoch/checkpoint on the trailing mean of the validation loss over this many epochs (1 = frozen behaviour)')
""")

# ---------------------------------------------------------------- 2. log target
sub1("""    props = np.array(props)
    mean, std = np.mean(props,axis=0,keepdims=True), np.std(props,axis=0,keepdims=True)""",
     """    props = np.array(props)
    if getattr(args, 'log_target', False):                                   # CAMPAIGN PATCH
        assert (props > 0).all(), 'log_target needs strictly positive targets'
        props = np.log10(props)
    mean, std = np.mean(props,axis=0,keepdims=True), np.std(props,axis=0,keepdims=True)""")

# ---------------------------------------------------------------- 3. stratified split construction
sub1("""def train(args, dags, graph, diffusion_args, props, norm_props, mol_feats, feat_lookup={}):
    dags_copy = deepcopy(dags)
    if args.test_seed != -1:
        random.seed(args.test_seed)
        all_idx = list(range(len(dags_copy)))
        random.shuffle(all_idx)
""",
     '''def stratified_order(strat, test_size, val_size, shuffled_positions):      # CAMPAIGN PATCH
    """Order the dataset positions as [train..., val..., test...] with every block holding the same
    proportion of each stratum, so the existing prefix/suffix slicing yields stratified splits.
    `shuffled_positions` is the already seed-shuffled list of positions (so the choice within a stratum is
    random and reproducible). Returns (order, effective_test_size)."""
    groups = {}
    for p in shuffled_positions:
        groups.setdefault(strat[p], []).append(p)
    keys = sorted(groups)
    n = len(shuffled_positions)
    # per-stratum test counts, then largest-remainder so the totals match round(test_size * n)
    want_test = int(round(test_size * n))
    exact = {k: test_size * len(groups[k]) for k in keys}
    cnt = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: -(exact[k] - cnt[k]))[:want_test - sum(cnt.values())]:
        cnt[k] += 1
    test, rest = [], {}
    for k in keys:
        g = groups[k]
        c = min(cnt[k], max(len(g) - 1, 0))          # never empty a stratum out of the training data
        test += g[len(g) - c:] if c else []
        rest[k] = g[:len(g) - c]
    n_rest = sum(len(rest[k]) for k in keys)
    want_val = int(round(val_size * n_rest))
    exact = {k: val_size * len(rest[k]) for k in keys}
    cnt = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: -(exact[k] - cnt[k]))[:want_val - sum(cnt.values())]:
        cnt[k] += 1
    val, tr = [], []
    for k in keys:
        g = rest[k]
        c = min(cnt[k], max(len(g) - 1, 0))
        val += g[len(g) - c:] if c else []
        tr += g[:len(g) - c]
    assert len(tr) + len(val) + len(test) == n
    # the caller slices train = order[:int((1-ts)*n)]; make that cut land exactly after train+val
    ts = (len(test) - 0.5) / n
    assert int((1 - ts) * n) == len(tr) + len(val), (ts, len(tr), len(val), len(test))
    return tr + val + test, ts


def train(args, dags, graph, diffusion_args, props, norm_props, mol_feats, feat_lookup={}):
    dags_copy = deepcopy(dags)
    if args.test_seed != -1:
        random.seed(args.test_seed)
        all_idx = list(range(len(dags_copy)))
        random.shuffle(all_idx)
        if getattr(args, 'stratify', 'none') != 'none':                      # CAMPAIGN PATCH
            labels = [int(x) for x in open(args.strata_file).read().split()]
            # dags_copy[p].dag_id is the walks_file line index of that polymer
            strat = [labels[d.dag_id] for d in dags_copy]
            if args.stratify == 'balanced':
                all_idx, args.test_size = stratified_order(strat, args.test_size, args.val_size, all_idx)
            else:                                                            # holdout: one stratum is the test split
                hs = int(args.holdout_stratum)
                inside = [p for p in all_idx if strat[p] == hs]
                outside = [p for p in all_idx if strat[p] != hs]
                assert 0 < len(inside) < len(all_idx), (hs, len(inside))
                all_idx = outside + inside
                args.test_size = (len(inside) - 0.5) / len(all_idx)
            with open(os.path.join(args.logs_folder, 'strata.txt'), 'w+') as f:
                f.write(' '.join(str(strat[p]) for p in all_idx) + '\\n')
''')

# ---------------------------------------------------------------- 4. loss / optimizer
sub1("""        opt = torch.optim.Adam(all_params, lr=args.lr)""",
     """        opt = torch.optim.Adam(all_params, lr=args.lr, weight_decay=getattr(args, 'weight_decay', 0.0))   # CAMPAIGN PATCH""")
sub1("""    loss_func = nn.MSELoss()   """,
     """    loss_func = nn.HuberLoss(delta=args.huber) if getattr(args, 'huber', 0.0) > 0 else nn.MSELoss()   # CAMPAIGN PATCH
    _logt = bool(getattr(args, 'log_target', False))                                                   # CAMPAIGN PATCH
    _inv = (lambda a: np.power(10.0, np.clip(a, -8.0, 9.0))) if _logt else (lambda a: a)
    _raw_tr = _inv(np.array([np.asarray(p, dtype=float) for p in props_train]))
    _raw_std = _raw_tr.std(axis=0)
    _sel_hist = []
""")

# ---------------------------------------------------------------- 5. smoothed checkpoint selection
sub1("""        if np.mean(loss_history) < best_loss:
            best_loss = np.mean(loss_history)
            print(f"best_loss epoch {epoch}", best_loss)""",
     """        _sel_hist.append(float(np.mean(loss_history)))                                   # CAMPAIGN PATCH
        _K = max(1, int(getattr(args, 'val_smooth', 1)))
        _sel_loss = float(np.mean(_sel_hist[-_K:]))
        if _sel_loss < best_loss:
            best_loss = _sel_loss
            print(f"best_loss epoch {epoch}", best_loss)""")

# ---------------------------------------------------------------- 6. extra metrics (val + heldout)
sub1("""        metrics.append(metric)
        df = pd.DataFrame(metrics)""",
     """        # ---- CAMPAIGN PATCH: transform-invariant and raw-space metrics, for validation and for the test split
        metric['val_loss_smooth'] = _sel_loss
        _blocks = [('', y, y_hat)]
        if heldout is not None and args.task == 'regression':
            _blocks.append(('heldout_', y_ho, y_ho_hat))
        for _tag, _yy, _yh in _blocks:
            for i in range(len(mean)):
                _nm = metric_names[i]
                try:
                    _rho = float(spearmanr(_yy[:, i], _yh[:, i]).correlation)
                except Exception:
                    _rho = float('nan')
                metric[f'{_tag}{_nm}_spearman'] = _rho
                _a = _inv(_yy[:, i] * std[i] + mean[i])
                _b = _inv(_yh[:, i] * std[i] + mean[i])
                _ok = np.isfinite(_a) & np.isfinite(_b)
                if _ok.sum() > 1:
                    metric[f'{_tag}raw_{_nm}_r^2'] = float(r2_score(_a[_ok], _b[_ok]))
                    metric[f'{_tag}raw_{_nm}_mae'] = float(np.abs(_b[_ok] - _a[_ok]).mean() / _raw_std[i])
                    metric[f'{_tag}raw_{_nm}_mse'] = float((((_b[_ok] - _a[_ok]) / _raw_std[i]) ** 2).mean())
        metrics.append(metric)
        df = pd.DataFrame(metrics)""")

# scipy import
sub1('''from sklearn.metrics import r2_score, roc_auc_score''', '''from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr   # CAMPAIGN PATCH""")

open(P, 'w').write(src)
print('patched', P)
