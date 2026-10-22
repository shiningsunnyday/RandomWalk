### Protocol (i): 1000 independent samples

| Method | Valid | Unique | Novel | Diversity | Median heavy atoms |
|---|---|---|---|---|---|
| Ours (motif-walk, i.i.d. sampling) | 56% | 7% | 100% | 0.90 | 20 |
| SMILES LSTM | 15% | 7% | 21% | 0.83 | 42 |
| SMILES LSTM (early-stopped) | 94% | 11% | 0.9% | 0.82 | 52 |
| SMILES transformer | 0.1% | 0.1% | 0% | 0.00 | 23 |
| SMILES transformer (early-stopped) | 87% | 11% | 0.2% | 0.82 | 52 |
| Hier-VAE | 100% | 22% | 54% | 0.87 | 46 |
| JT-VAE | 100% | 22% | 92% | 0.79 | 58 |
| Graph GA (unconditional) | 100% | 91% | 100% | 0.91 | 25 |
| Graph GA (CO2-conditioned) | 100% | 86% | 98% | 0.83 | 43 |
| REINVENT 4 (transfer only) | 88% | 46% | 57% | 0.88 | 40 |
| REINVENT 4 (transfer + RL) | 99% | 73% | 99% | 0.87 | 24 |
| DEG (grammar induction only) | 100% | 50% | 60% | 0.82 | 54 |

### Protocol (ii): sample until 1000 distinct molecules

| Method | Samples drawn | Reached | Distinct found | Novel | Diversity |
|---|---|---|---|---|---|
| Ours (motif-walk, i.i.d. sampling) | 10,000 | no | 116 | 100% | 0.89 |
| SMILES LSTM | 20,000 | no | 471 | 83% | 0.85 |
| SMILES LSTM (early-stopped) | 20,000 | no | 149 | 26% | 0.83 |
| SMILES transformer | 20,000 | no | 21 | 90% | 0.86 |
| SMILES transformer (early-stopped) | 20,000 | no | 144 | 24% | 0.83 |
| Hier-VAE | 1,000 | no | 219 | 82% | 0.87 |
| JT-VAE | 1,000 | no | 221 | 99% | 0.79 |
| Graph GA (unconditional) | 1,104 | yes | 1,000 | 100% | 0.91 |
| Graph GA (CO2-conditioned) | 1,000 | no | 857 | 99% | 0.83 |
| REINVENT 4 (transfer only) | 2,441 | yes | 1,000 | 93% | 0.89 |
| REINVENT 4 (transfer + RL) | 1,419 | yes | 1,000 | 100% | 0.87 |
| DEG (grammar induction only) | 1,000 | no | 503 | 98% | 0.82 |
