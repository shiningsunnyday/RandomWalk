"""
Character-level autoregressive SMILES language models on the 114 polymer repeat units:
  --arch lstm        : 2-layer LSTM, embed 64, hidden 256  (the classic SMILES-LSTM / ChemTS-style LM)
  --arch transformer : 4-layer causal transformer decoder, d_model 256, 8 heads (SMILES transformer)

Protocol: 100 molecules train / 14 held out, early stopping on held-out NLL (patience 200 epochs,
max 3000), sampling from the best-held-out-NLL checkpoint. Novelty is still measured against all 114.
Both the training and the held-out NLL are logged so the overfitting at n=114 is visible in the log.
Sampling is multinomial at temperature 1.0 with no validity filtering and no retries, so the Valid
metric is the model's own.

usage (env polymer_walk, GPU):
  python common/smiles_lm.py --arch lstm --out runs/smiles_lstm --nsamples 1000
"""
import os, sys, json, math, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(HERE, 'common', 'train_smiles.txt')
BOS, EOS, PAD = 0, 1, 2


class LSTMLM(nn.Module):
    def __init__(self, V, embed=64, hidden=256, layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(V, embed, padding_idx=PAD)
        self.rnn = nn.LSTM(embed, hidden, layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden, V)

    def forward(self, x, state=None):
        h, state = self.rnn(self.emb(x), state)
        return self.out(h), state


class TransformerLM(nn.Module):
    def __init__(self, V, d=256, nhead=8, layers=4, dropout=0.2, maxlen=512):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, V)

    def forward(self, x, state=None):
        T = x.shape[1]
        p = torch.arange(T, device=x.device)[None, :]
        h = self.emb(x) + self.pos(p)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        h = self.tr(h, mask=mask, is_causal=True)
        return self.out(h), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['lstm', 'transformer'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--nsamples', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--patience', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--fullfit', action='store_true',
                    help='train on all 114 for exactly --epochs epochs, no early stopping, sample '
                         'from the final model (the memorisation regime)')
    ap.add_argument('--seed', type=int, default=0)
    # Unified-protocol resampling (BRIEFING_2 table B): load the checkpoint this run already trained and draw more
    # samples, so protocol (ii) "sample until N unique" costs sampling only and the model is bit-identical to the
    # one protocol (i) used.
    ap.add_argument('--sample_only', action='store_true')
    ap.add_argument('--samples_name', default='samples.txt')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda:0'

    smis = [l.strip() for l in open(TRAIN) if l.strip()]
    chars = sorted({c for s in smis for c in s})
    stoi = {c: i + 3 for i, c in enumerate(chars)}
    itos = {i + 3: c for i, c in enumerate(chars)}
    V = len(chars) + 3
    maxlen = max(len(s) for s in smis) + 2
    print(f'{len(smis)} SMILES, {len(chars)} characters, vocab {V}, max length {maxlen}', flush=True)

    def enc(s):
        return [BOS] + [stoi[c] for c in s] + [EOS]

    idx = list(range(len(smis)))
    random.Random(0).shuffle(idx)
    tr_idx, va_idx = idx[:100], idx[100:]
    if a.fullfit:
        tr_idx, va_idx = list(range(len(smis))), idx[100:]   # val set is a subset of train here

    def batchify(ids):
        seqs = [enc(smis[i]) for i in ids]
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), PAD, dtype=torch.long)
        for k, s in enumerate(seqs):
            x[k, :len(s)] = torch.tensor(s)
        return x.to(dev)

    model = (LSTMLM(V) if a.arch == 'lstm' else TransformerLM(V, maxlen=maxlen + 64)).to(dev)
    print('params', sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def nll(ids, train):
        model.train(train)
        tot, n = 0.0, 0
        order = list(ids)
        if train:
            random.shuffle(order)
        for i in range(0, len(order), a.batch_size):
            x = batchify(order[i:i + a.batch_size])
            logits, _ = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1),
                                   ignore_index=PAD, reduction='sum')
            ntok = (x[:, 1:] != PAD).sum()
            if train:
                opt.zero_grad(); (loss / ntok).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); n += ntok.item()
        return tot / n

    best, best_ep, bad = 1e9, -1, 0
    ckpt = os.path.join(a.out, 'model.pt')
    t0 = time.time()
    if a.sample_only:
        assert os.path.exists(ckpt), ckpt
        best_ep, best, train_time = -1, float('nan'), 0.0
    hist = []
    for ep in range(0 if a.sample_only else a.epochs):
        trl = nll(tr_idx, True)
        with torch.no_grad():
            val = nll(va_idx, False)
        hist.append((ep, trl, val))
        if a.fullfit:
            torch.save(model.state_dict(), ckpt)
            best, best_ep = val, ep
            if ep % 25 == 0:
                print(f'ep {ep} train_nll {trl:.4f} (fullfit, no early stopping) '
                      f'({time.time()-t0:.0f}s)', flush=True)
            continue
        if val < best - 1e-4:
            best, best_ep, bad = val, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        if ep % 25 == 0 or bad == 0:
            print(f'ep {ep} train_nll {trl:.4f} val_nll {val:.4f} best {best:.4f}@{best_ep} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        if bad >= a.patience:
            print(f'early stop at epoch {ep}', flush=True)
            break
    if not a.sample_only:
        train_time = time.time() - t0
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    with torch.no_grad():
        final_tr = nll(tr_idx, False)
    print(f'best checkpoint: epoch {best_ep}, train_nll {final_tr:.4f}, val_nll {best:.4f}', flush=True)

    # sampling
    out = []
    t1 = time.time()
    with torch.no_grad():
        B = 100
        while len(out) < a.nsamples:
            x = torch.full((B, 1), BOS, dtype=torch.long, device=dev)
            done = torch.zeros(B, dtype=torch.bool, device=dev)
            toks = [[] for _ in range(B)]
            state = None
            cur = x
            for t in range(maxlen + 20):
                if a.arch == 'lstm':
                    logits, state = model(cur, state)
                    logits = logits[:, -1]
                else:
                    logits, _ = model(x)
                    logits = logits[:, -1]
                logits[:, PAD] = -1e9
                if t == 0:
                    logits[:, EOS] = -1e9
                p = F.softmax(logits / a.temperature, dim=-1)
                nxt = torch.multinomial(p, 1)
                for k in range(B):
                    if not done[k]:
                        tk = nxt[k, 0].item()
                        if tk == EOS:
                            done[k] = True
                        else:
                            toks[k].append(tk)
                x = torch.cat([x, nxt], 1)
                cur = nxt
                if done.all():
                    break
            out += [''.join(itos.get(t, '') for t in tk) for tk in toks]
    out = out[:a.nsamples]
    open(os.path.join(a.out, a.samples_name), 'w').write('\n'.join(s if s else '<empty>' for s in out) + '\n')
    json.dump(dict(arch=a.arch, params=sum(p.numel() for p in model.parameters()),
                   best_epoch=best_ep, best_val_nll=best, train_nll_at_best=final_tr,
                   n_train=len(tr_idx), n_val=len(va_idx), vocab=V, maxlen=maxlen,
                   train_seconds=train_time, sample_seconds=time.time() - t1, argv=vars(a),
                   nll_history=hist[-5:]),
              open(os.path.join(a.out, 'run_info.json'), 'w'), indent=1)
    print('wrote', len(out), 'samples', flush=True)


if __name__ == '__main__':
    main()
