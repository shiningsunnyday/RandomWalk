"""
Dataset-bias characterization of the 114 labelled polymers, and the stratum label files the campaign's
stratified / cluster-holdout splits use.

Answers Reviewer 1's "no rigorous evaluation of dataset bias is performed" directly:
  * property distributions (range, skew, how much of the spread sits in a handful of polymers)
  * backbone family / name-prefix composition
  * motif-composition structure: how many of the 99 motifs are used, how concentrated the usage is
  * fingerprint-space coverage: Tanimoto nearest-neighbour distribution, how much of the space is one cluster
  * Butina/KMeans clusters and Murcko-scaffold groups, with per-cluster property statistics

Outputs (all under campaign/analysis/):
  dataset_bias.json              every number quoted in RESULTS.md / the SI paragraph
  strata_y.txt                   3 strata: tertiles of log10 H2 permeability      (label-aware)
  strata_struct.txt              3 strata: tertiles of the number of motifs        (label-free)
  strata_fpclust.txt             K strata: KMeans clusters of the polymer Morgan fingerprint (label-free)
  strata_scaffold.txt            K strata: grouped Bemis-Murcko scaffolds          (label-free)
  figs/bias_*.png                SI-candidate figures
Each strata file has one integer per walks_file line (114 lines, in file order), whitespace separated.

usage: python analysis/dataset_bias.py
"""
import os, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
RDLogger.DisableLog('rdApp.*')

A = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(A)                       # campaign/
GC = os.path.join(R, '..', 'data', 'group-contrib')
F = os.path.join(A, 'figs'); os.makedirs(F, exist_ok=True)
GASES = ['H2', 'N2', 'O2', 'CH4', 'CO2']
PAIRS = [('H2', 'N2'), ('O2', 'N2'), ('CO2', 'CH4')]

lines = [l.rstrip('\n') for l in open(os.path.join(GC, 'polymer_walks_v2_preprocess.txt')) if l.strip()]
names = [l.split()[0] for l in lines]
walks = [l.split()[1] for l in lines]
def _f(x):
    x = x.strip()
    return float('nan') if x in ('-', '_', '') else float(x)
props = np.array([[_f(x) for x in l.split()[-1].strip('()').split(',')] for l in lines])   # H2 N2 O2 CH4 CO2
out_missing = {g: int(np.isnan(props[:, j]).sum()) for j, g in enumerate(['H2', 'N2', 'O2', 'CH4', 'CO2'])}
smi = [l.split()[0] for l in open(os.path.join(GC, 'polymer_walks_v2_smiles.txt')) if l.strip()]
n = len(lines)
assert n == len(smi) == 114, (n, len(smi))

out = {'n_polymers': n}

# ---------------------------------------------------------------- 1. property distributions
pdist = {}
for j, g in enumerate(GASES):
    v = props[:, j]; v = v[np.isfinite(v)]
    pdist[g] = dict(n=int(len(v)), min=float(v.min()), max=float(v.max()), median=float(np.median(v)),
                    mean=float(v.mean()), std=float(v.std()), log10_span=float(np.log10(v.max() / v.min())),
                    skew=float(((v - v.mean()) ** 3).mean() / v.std() ** 3),
                    # how concentrated is the variance: share of total squared deviation in the top 5 polymers
                    top5_var_share=float(np.sort((v - v.mean()) ** 2)[-5:].sum() / ((v - v.mean()) ** 2).sum()))
for a, b in PAIRS:
    v = props[:, GASES.index(a)] / props[:, GASES.index(b)]; v = v[np.isfinite(v)]
    pdist[f'{a}/{b}'] = dict(n=int(len(v)), min=float(v.min()), max=float(v.max()), median=float(np.median(v)),
                             log10_span=float(np.log10(v.max() / v.min())),
                             skew=float(((v - v.mean()) ** 3).mean() / v.std() ** 3),
                             top5_var_share=float(np.sort((v - v.mean()) ** 2)[-5:].sum() / ((v - v.mean()) ** 2).sum()))
out['property_distributions'] = pdist
out['missing_values_per_gas'] = out_missing

# ---------------------------------------------------------------- 2. families from the polymer names
def family(nm):
    s = nm.upper()
    if 'PIM' in s and 'PI' not in s.replace('PIM', ''): base = 'PIM'
    elif 'PIM-PI' in s or ('PIM' in s and 'PI' in s): base = 'PIM-PI'
    elif 'KAUST' in s: base = 'KAUST-PI'
    elif re.match(r'^(TPDA|TBDA|SPDA|PMDA|BPDA|TDA|ODA|BTDA|BPADA|SBIDA|SBFDA|PPIMDA|EAD)', s): base = 'polyimide (dianhydride series)'
    elif 'PI' in s: base = 'polyimide (other)'
    elif s[0].isdigit() or s[0] == '-': base = 'numbered entry'
    else: base = 'other'
    return base

fams = [family(x) for x in names]
famcount = {k: int(sum(f == k for f in fams)) for k in sorted(set(fams))}
out['families'] = famcount
out['family_largest_share'] = max(famcount.values()) / n

# ---------------------------------------------------------------- 3. motif composition
motif_re = re.compile(r'([PSL]\d+)')
comp = []
allm = set()
for w in walks:
    ms = motif_re.findall(w)
    comp.append(ms); allm |= set(ms)
allm = sorted(allm, key=lambda s: (s[0], int(s[1:])))
M = np.zeros((n, len(allm)))
for i, ms in enumerate(comp):
    for m in ms: M[i, allm.index(m)] += 1
usage = M.sum(0)
order = np.argsort(-usage)
out['motifs'] = dict(n_motifs_in_vocab=99, n_motifs_used=len(allm),
                     median_motifs_per_polymer=float(np.median(M.sum(1))),
                     mean_motifs_per_polymer=float(M.sum(1).mean()),
                     min_motifs=int(M.sum(1).min()), max_motifs=int(M.sum(1).max()),
                     top10=[[allm[i], int(usage[i]), float(usage[i] / usage.sum())] for i in order[:10]],
                     share_of_motif_slots_in_top10=float(usage[order[:10]].sum() / usage.sum()),
                     motifs_used_once=int((usage == 1).sum()),
                     polymers_containing_S32=int((M[:, allm.index('S32')] > 0).sum()) if 'S32' in allm else 0,
                     polymers_containing_S20=int((M[:, allm.index('S20')] > 0).sum()) if 'S20' in allm else 0)

# ---------------------------------------------------------------- 4. fingerprint-space coverage
mols = [Chem.MolFromSmiles(s) for s in smi]
assert all(m is not None for m in mols)
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in mols]
S = np.eye(n)
for i in range(n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    S[i] = sims
np.fill_diagonal(S, 0.0)
nn = S.max(1)
out['fingerprint_space'] = dict(
    mean_pairwise_tanimoto=float(S[np.triu_indices(n, 1)].mean()),
    median_pairwise_tanimoto=float(np.median(S[np.triu_indices(n, 1)])),
    median_nearest_neighbour_tanimoto=float(np.median(nn)),
    frac_with_nn_above_0p7=float((nn > 0.7).mean()),
    frac_with_nn_above_0p5=float((nn > 0.5).mean()),
    internal_diversity=float(1 - S[np.triu_indices(n, 1)].mean()))

# Butina-style clustering on the Tanimoto distance, plus KMeans on the fingerprint bits
X = np.array([np.array(f, dtype=float) for f in fps])
Xb = X[:, X.std(0) > 0]
K = 4
km = KMeans(n_clusters=K, n_init=20, random_state=0).fit(Xb)
fpclust = km.labels_
out['fp_clusters'] = {str(k): int((fpclust == k).sum()) for k in range(K)}
out['fp_cluster_largest_share'] = float(max(out['fp_clusters'].values()) / n)

scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m.GetNumAtoms() else '' for m in mols]
uniq_scaf = {}
for s in scaf: uniq_scaf[s] = uniq_scaf.get(s, 0) + 1
out['scaffolds'] = dict(n_unique=len(uniq_scaf), largest=int(max(uniq_scaf.values())),
                        singletons=int(sum(1 for v in uniq_scaf.values() if v == 1)))
# group scaffolds into K strata, biggest scaffolds first, so each stratum is a set of whole scaffolds
sc_order = sorted(uniq_scaf, key=lambda s: -uniq_scaf[s])
sc_lab, load = {}, [0] * K
for s in sc_order:
    k = int(np.argmin(load)); sc_lab[s] = k; load[k] += uniq_scaf[s]
scaf_strata = np.array([sc_lab[s] for s in scaf])
out['scaffold_strata_sizes'] = {str(k): int((scaf_strata == k).sum()) for k in range(K)}

# ---------------------------------------------------------------- 5. per-cluster property statistics
def cluster_stats(lab):
    d = {}
    for k in sorted(set(lab.tolist())):
        m = lab == k
        d[str(int(k))] = dict(n=int(m.sum()),
                              median_H2=float(np.nanmedian(props[m, 0])),
                              median_O2=float(np.nanmedian(props[m, 2])),
                              median_CO2=float(np.nanmedian(props[m, 4])),
                              median_O2_N2=float(np.nanmedian(props[m, 2] / props[m, 1])),
                              log10_H2_mean=float(np.nanmean(np.log10(props[m, 0]))),
                              log10_H2_std=float(np.nanstd(np.log10(props[m, 0]))))
    return d
out['fp_cluster_properties'] = cluster_stats(fpclust)
out['scaffold_strata_properties'] = cluster_stats(scaf_strata)
out['family_properties'] = cluster_stats(np.array([sorted(famcount).index(f) for f in fams]))
out['family_order'] = sorted(famcount)

# ---------------------------------------------------------------- 6. strata files
def tertiles(v):
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    q = np.quantile(v[ok], [1 / 3, 2 / 3])
    lab = np.digitize(v, q)
    lab[~ok] = 1                                                  # unlabelled for this gas: never in that subset
    return lab

strata = {
    'strata_y_01.txt': tertiles(np.log10(props[:, 0])),           # label-aware: log H2 permeability tertile
    'strata_y_21.txt': tertiles(np.log10(props[:, 2])),           # label-aware: log O2 permeability tertile
    'strata_y_43.txt': tertiles(np.log10(props[:, 4])),           # label-aware: log CO2 permeability tertile
    'strata_struct.txt': tertiles(M.sum(1)),                      # label-free: motif-count tertile
    'strata_fpclust.txt': fpclust,                                # label-free: fingerprint KMeans
    'strata_scaffold.txt': scaf_strata,                           # label-free: grouped Murcko scaffolds
}
for fn, lab in strata.items():
    open(os.path.join(A, fn), 'w').write(' '.join(str(int(x)) for x in lab) + '\n')
    out.setdefault('strata_sizes', {})[fn] = {str(int(k)): int((np.asarray(lab) == k).sum()) for k in sorted(set(np.asarray(lab).tolist()))}

# ---------------------------------------------------------------- 7. figures
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, k in zip(axes.ravel(), ['H2', 'O2', 'CO2', 'H2/N2', 'O2/N2', 'CO2/CH4']):
    if '/' in k:
        a, b = k.split('/'); v = props[:, GASES.index(a)] / props[:, GASES.index(b)]
    else:
        v = props[:, GASES.index(k)]
    v = v[np.isfinite(v)]
    ax.hist(np.log10(v), bins=22, color='#4c78a8')
    ax.set_xlabel(f'log10 {k}' + (' permeability (barrer)' if '/' not in k else ' selectivity'))
    ax.set_ylabel('polymers')
    ax.set_title(f'span {np.log10(v.max()/v.min()):.1f} decades', fontsize=10)
fig.suptitle('Property distributions of the 114 labelled polymers', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(F, 'bias_property_hist.png'), dpi=200); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].hist(nn, bins=20, color='#59a14f'); axes[0].set_xlabel('Tanimoto to nearest neighbour (Morgan r=3, 2048)')
axes[0].set_ylabel('polymers'); axes[0].set_title(f'median {np.median(nn):.2f}; {100*(nn>0.7).mean():.0f}% above 0.7', fontsize=10)
axes[1].hist(S[np.triu_indices(n, 1)], bins=40, color='#e15759'); axes[1].set_xlabel('pairwise Tanimoto')
axes[1].set_ylabel('pairs'); axes[1].set_title(f'mean {S[np.triu_indices(n,1)].mean():.2f}', fontsize=10)
pc = PCA(2).fit_transform(Xb)
for k in range(K):
    m = fpclust == k
    axes[2].scatter(pc[m, 0], pc[m, 1], s=28, label=f'cluster {k} (n={m.sum()})')
axes[2].set_xlabel('PC1 of fingerprint'); axes[2].set_ylabel('PC2'); axes[2].legend(fontsize=8)
axes[2].set_title('fingerprint-space clusters', fontsize=10)
fig.suptitle('Chemical-space coverage of the 114 labelled polymers', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(F, 'bias_chemspace.png'), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(20), usage[order[:20]], color='#9c755f')
ax.set_xticks(range(20)); ax.set_xticklabels([allm[i] for i in order[:20]], rotation=60, fontsize=8)
ax.set_ylabel('occurrences in the 114 walks')
ax.set_title(f'{len(allm)} of 99 motifs appear; top 10 cover '
             f'{100*usage[order[:10]].sum()/usage.sum():.0f}% of all motif slots', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(F, 'bias_motif_usage.png'), dpi=200); plt.close(fig)

json.dump(out, open(os.path.join(A, 'dataset_bias.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('fp_cluster_properties', 'scaffold_strata_properties', 'family_properties')}, indent=1))
