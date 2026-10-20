"""
Generation baselines -> baselines/outputs/si_gen_table.json (SI Table S14) and the Retro*-vs-size statistics.

Reads generation/metrics.jsonl (written by the generation-baseline drivers, which implement the same metric
definitions as compute_numbers.py lines 18-37) and our own Table 1 numbers from outputs/numbers.json, and emits the
table rows plus the correlation between Retro* success and molecule size. That correlation is the reason the table
carries the 114 dataset monomers as a row: they are real, experimentally characterised membrane polymers, and they
score far below our generated structures on Retro*.

usage: python bl_gen_si.py
"""
import os, sys, json
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
RV = os.path.dirname(H)
GEN = os.path.join(H, 'generation', 'metrics.jsonl')

LABELS = [                                   # (metrics.jsonl name, table label); order is the table order
    ('dataset_114', 'The 114 dataset monomers (reference)'),
    ('graph_ga_uncond', 'Graph GA (unconditional)'),
    ('graph_ga_co2', 'Graph GA (CO₂-conditioned)'),
    ('reinvent_rl', 'REINVENT 4 (transfer + RL)'),
    ('reinvent_tl', 'REINVENT 4 (transfer only)'),
    ('hiervae', 'Hier-VAE'),
    ('jtvae', 'JT-VAE'),
    ('deg', 'DEG (grammar induction only)'),
    ('smiles_lstm_fullfit', 'SMILES LSTM'),
    ('smiles_lstm', 'SMILES LSTM (early-stopped)'),
    ('smiles_transformer_fullfit', 'SMILES transformer'),
    ('smiles_transformer', 'SMILES transformer (early-stopped)'),
]


def pct(x):
    if x is None:
        return '—'
    v = 100 * x
    return f'{v:.1f}%' if 0 < v < 1 else f'{v:.0f}%'   # a memorising LM's 0.4% novel must not round away to 0%


def main():
    N = json.load(open(os.path.join(RV, 'outputs', 'numbers.json')))['table1_exact']
    rows_in = {}
    if os.path.exists(GEN):
        for line in open(GEN):
            r = json.loads(line)
            rows_in[r['name']] = r                     # a later row for the same name supersedes an earlier one

    ours = {'label': 'Ours (motif-walk, best-first)', 'valid': N['valid_smiles'], 'unique': N['unique'],
            'novel': N['novel'], 'diversity': N['diversity'], 'retro': N['retro'], 'retro_n': N['n_gen'],
            'median_heavy_atoms': None}
    # our own median heavy-atom count, from the clean SMILES behind Table 1
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    ha = [Chem.MolFromSmiles(l.split(' ')[0]).GetNumHeavyAtoms()
          for l in open(os.path.join(RV, 'outputs', 'gen', 'bestfirst_n1000_clean.txt')) if l.strip()]
    ours['median_heavy_atoms'] = float(np.median(ha))

    table, pts, missing = [], [], []
    def add(label, r):
        table.append([label, pct(r.get('valid')), pct(r.get('unique')), pct(r.get('novel')),
                      '—' if r.get('diversity') is None else f"{r['diversity']:.2f}",
                      pct(r.get('retro')),
                      '—' if r.get('median_heavy_atoms') is None else f"{r['median_heavy_atoms']:.0f}"])
        if r.get('retro') is not None and r.get('median_heavy_atoms') is not None:
            pts.append((label, float(r['median_heavy_atoms']), float(r['retro'])))

    add(ours['label'], ours)
    for name, label in LABELS:
        if name in rows_in:
            add(label, rows_in[name])
        else:
            missing.append(label)

    out = {'header': ['Method', 'Valid', 'Unique', 'Novel', 'Diversity', 'Retro*', 'Median heavy atoms'],
           'rows': table, 'missing': missing}
    if len(pts) >= 4:
        from scipy.stats import pearsonr, spearmanr
        x = np.array([p[1] for p in pts]); y = np.array([p[2] for p in pts])
        r, pv = pearsonr(x, y); rs, ps = spearmanr(x, y)
        out['size_vs_retro'] = {'n': len(pts), 'pearson_r': round(float(r), 3), 'pearson_p': float(pv),
                                'spearman_rho': round(float(rs), 3), 'spearman_p': float(ps),
                                'points': [{'method': a, 'median_heavy_atoms': b, 'retro': c} for a, b, c in pts],
                                'r_text': f'{r:.2f}'.replace('-', '−'),
                                'dataset_retro': pct(rows_in.get('dataset_114', {}).get('retro')),
                                'ours_retro': pct(ours['retro']),
                                'ours_median_ha': f"{ours['median_heavy_atoms']:.0f}",
                                'dataset_median_ha': ('—' if 'dataset_114' not in rows_in else
                                                      f"{rows_in['dataset_114']['median_heavy_atoms']:.0f}")}
    json.dump(out, open(os.path.join(H, 'outputs', 'si_gen_table.json'), 'w'), indent=1)
    print(f"{len(table)} rows; still missing: {missing or 'none'}")
    if 'size_vs_retro' in out:
        s = out['size_vs_retro']
        print(f"Retro* vs median heavy atoms: Pearson r = {s['pearson_r']} (p = {s['pearson_p']:.4f}), "
              f"Spearman rho = {s['spearman_rho']} (p = {s['spearman_p']:.4f}), n = {s['n']}")
    for row in table:
        print('  ' + ' | '.join(row))


if __name__ == '__main__':
    main()
