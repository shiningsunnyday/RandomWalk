"""
JT-VAE baseline (Jin et al., ICML 2018) on the 114 polymer repeat units.

Code: the JT-VAE port shipped in ~/torch-molecule
(`torch_molecule.generator.jtvae`, which is the kamikaze0923/jtvae re-implementation of the original
ICML'18 model). The repo's top-level `torch_molecule/__init__.py` cannot be imported as-is here --
`from .generator.random_walk import RandomWalkMolecularGenerator` raises
`NameError: name 'L_grammar' is not defined` inside that (unrelated, in-development) module -- so that
one submodule is stubbed in sys.modules before the import. Nothing in ~/torch-molecule is modified.

Sampling: `JTNNVAE.sample_prior` draws z ~ N(0, I) for the tree and the graph latent and decodes,
one molecule per call, no validity filtering and no retries; a decoder exception is recorded as an
empty line so it counts as invalid in the metrics.

usage (env torch_molecule, GPU):
  python common/jtvae_run.py --out runs/jtvae --epochs 400 --nsamples 1000
"""
import os, sys, json, time, types, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TM = '/home/msun415/torch-molecule'
sys.path.insert(0, TM)
stub = types.ModuleType('torch_molecule.generator.random_walk')
stub.RandomWalkMolecularGenerator = object
stub.RandomWalkMoleculeGenerator = object
sys.modules['torch_molecule.generator.random_walk'] = stub
from torch_molecule.generator.jtvae.modeling_jtvae import JTVAEMolecularGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--nsamples', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--hidden_size', type=int, default=450)
    ap.add_argument('--latent_size', type=int, default=56)
    ap.add_argument('--max_seconds', type=float, default=5400)
    ap.add_argument('--seed', type=int, default=0)
    # sample-only resampling for the unified protocol: load the checkpoint this run trained and draw more
    ap.add_argument('--sample_only', action='store_true')
    ap.add_argument('--samples_name', default='samples.txt')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    smis = [l.strip() for l in open(os.path.join(HERE, 'common', 'train_smiles.txt')) if l.strip()]
    print(f'training JT-VAE on {len(smis)} SMILES for {a.epochs} epochs', flush=True)
    gen = JTVAEMolecularGenerator(hidden_size=a.hidden_size, latent_size=a.latent_size,
                                  batch_size=a.batch_size, epochs=a.epochs,
                                  verbose=False, device='cuda:0')
    # fit() inlined from JTVAEMolecularGenerator.fit so that (a) training can be stopped on a wall
    # clock budget and still sample, and (b) the per-epoch loss is logged as it goes.
    from torch_molecule.generator.jtvae.jtnn.vocab import Vocab
    from torch_molecule.generator.jtvae.jtnn.datautils import MolTreeFolder
    t0 = time.time()
    X_train, _ = gen._validate_inputs(smis, None, num_task=0, return_rdkit_mol=False)
    gen.vocab = Vocab(gen._extract_vocab(X_train))
    gen._initialize_model(gen.model_class)
    gen.model.initialize_parameters()
    optimizer, scheduler = gen._setup_optimizers()
    train_dataset = [x for x in gen._convert_to_tensor(X_train) if x is not None]
    loader = MolTreeFolder(train_dataset, gen.vocab, gen.batch_size, num_workers=0)
    step_len = len(X_train) // gen.batch_size
    print(f'setup (vocab {len(gen.vocab.vocab)} + tensorize {len(train_dataset)} mols): '
          f'{time.time()-t0:.0f}s', flush=True)
    gen.fitting_loss, total_step = [], 0
    t_train = time.time()
    if a.sample_only:
        ck = torch.load(os.path.join(a.out, 'model.pt'), map_location=gen.device)
        gen.model.load_state_dict(ck['state'])
        gen.fitting_loss, epoch = [float('nan')], -1
    for epoch in range(0 if a.sample_only else a.epochs):
        losses, total_step = gen._train_epoch(loader, optimizer, scheduler, epoch, total_step, step_len)
        gen.fitting_loss.append(float(np.mean(losses)))
        if epoch % 5 == 0 or epoch == a.epochs - 1:
            print(f'epoch {epoch} loss {gen.fitting_loss[-1]:.4f} ({time.time()-t_train:.0f}s)', flush=True)
        if time.time() - t_train > a.max_seconds:
            print(f'wall-clock budget {a.max_seconds}s reached at epoch {epoch}', flush=True)
            break
    gen.fitting_epoch = epoch
    gen.is_fitted_ = True
    tt = time.time() - t0
    if a.sample_only:
        print(f'sample_only: loaded {os.path.join(a.out, "model.pt")}', flush=True)
    print(f'trained {epoch+1} epochs in {tt:.0f}s; vocab size {len(gen.vocab.vocab)}; '
          f'final epoch loss {gen.fitting_loss[-1]:.4f}', flush=True)
    if not a.sample_only:
        torch.save(dict(state=gen.model.state_dict(), vocab=list(gen.vocab.vocab)),
                   os.path.join(a.out, 'model.pt'))
        json.dump(dict(fitting_loss=gen.fitting_loss), open(os.path.join(a.out, 'loss.json'), 'w'))

    gen.model.eval()
    out, errs = [], 0
    t1 = time.time()
    with torch.no_grad():
        while len(out) < a.nsamples:
            try:
                s = gen.model.sample_prior(gen.device)
            except Exception as e:
                s, errs = '', 1 + errs
            if isinstance(s, (list, tuple)):
                s = s[0] if s else ''
            out.append(s if isinstance(s, str) and s else '<empty>')
            if len(out) % 100 == 0:
                print(f'{len(out)} sampled, {errs} decoder exceptions, {time.time()-t1:.0f}s', flush=True)
    open(os.path.join(a.out, a.samples_name), 'w').write('\n'.join(out) + '\n')
    json.dump(dict(epochs=a.epochs, epochs_run=gen.fitting_epoch + 1, train_seconds=tt, sample_seconds=time.time() - t1,
                   decoder_exceptions=errs, vocab_size=len(gen.vocab.vocab),
                   final_loss=gen.fitting_loss[-1], argv=vars(a)),
              open(os.path.join(a.out, 'run_info.json'), 'w'), indent=1)
    print('wrote', len(out), 'samples;', errs, 'decoder exceptions', flush=True)


if __name__ == '__main__':
    main()
