"""
The train / validation / test split of a predictor run, obtained safely.

`test_idx.txt` in the FROZEN results_v2/outputs/pred/* dirs is wrong: those runs were produced before the
snapshot's main.py fix landed, by `idx_partition(all_idx, all_idx, ...)`, which indexes the shuffled
permutation by itself. The file therefore holds [all_idx[i] for i in all_idx[int(0.8N):]] instead of
all_idx[int(0.8N):]. `val_idx.txt`, `mean_and_std.txt` and every `heldout_*` metric in those dirs are correct;
only `test_idx.txt` is affected. Verified: the frozen file equals the doubly-indexed permutation exactly, and
shares 3-4 indices with val_idx.txt, which a real split cannot do. Campaign runs write the correct indices.

test_positions() reads the file, detects the bug by its overlap with val_idx.txt, and falls back to
recomputing the split from the seed, which is fully deterministic. It always asserts disjointness.
Positions are indices into the gas pair's labelled subset in walks_file line order (= dags_copy order).
"""
import os, random
import numpy as np


def subset_lines(props, a, b):
    """walks_file line indices with both properties of the pair present, in line order"""
    return [i for i in range(len(props)) if np.isfinite(props[i, a]) and np.isfinite(props[i, b])]


def split_idx(n, seed, test_size=0.2, val_size=0.2, train_size=1.0):
    """main.py's split for --test_seed seed --train_size 1.0 --test_size 0.2 --val_size 0.2"""
    a = list(range(n)); random.seed(seed); random.shuffle(a)
    k = int(min(train_size, 1 - test_size) * n)
    train, test = a[:k], a[k:]
    nv = int(round(val_size * len(train)))
    return train[:-nv], train[-nv:], test


def test_positions(d, n, seed, stratify='none'):
    """test-split positions for run dir `d`; raises if neither source can be trusted"""
    val = [int(x) for x in open(os.path.join(d, 'val_idx.txt'))]
    f = os.path.join(d, 'test_idx.txt')
    test = [int(x) for x in open(f)] if os.path.exists(f) else []
    if test and not (set(test) & set(val)):
        src = 'file'
    else:
        assert stratify in (None, '', 'none'), \
            f'{d}: test_idx.txt is the buggy doubly-indexed one and the split is stratified, so it cannot be ' \
            f'recomputed from the seed alone'
        _, va, test = split_idx(n, seed)
        assert sorted(va) == sorted(val), f'{d}: recomputed validation split does not match val_idx.txt'
        src = 'recomputed'
    assert not (set(test) & set(val)), f'{d}: test and validation splits overlap'
    return test, src
