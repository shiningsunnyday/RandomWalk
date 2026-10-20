from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import fcntl
import numpy as np 
import time
import os
import shutil
import argparse
import json
from pathlib import Path
import random
import multiprocessing as mp
from itertools import product
from tqdm import tqdm
from moses.metrics import FCDMetric, ScafMetric, compute_intermediate_statistics, get_all_metrics
from moses.utils import mapper, get_mol
from eden.graph import vectorize
from sklearn.metrics.pairwise import pairwise_kernels
import networkx as nx


def lock(f):
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        return False
    return True


class InternalDiversity():
    def distance(self, mol1, mol2, dtype="Tanimoto"):
        assert dtype in ["Tanimoto"]
        if dtype == "Tanimoto":
            sim = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2))
            return 1 - sim
        else:
            raise NotImplementedError

    def get_diversity(self, mol_list, dtype="Tanimoto"):
        similarity = 0
        mol_list = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mol_list] 
        for i in range(len(mol_list)):
            sims = DataStructs.BulkTanimotoSimilarity(mol_list[i], mol_list[:i])
            similarity += sum(sims)
        n = len(mol_list)
        n_pairs = n * (n - 1) / 2
        diversity = 1 - similarity / n_pairs
        return diversity
    

def kernel_compute(X, Y=None, n_jobs=None):
    X = vectorize(X, complexity=4, discrete=True)
    if Y is not None:
        Y = vectorize(Y, complexity=4, discrete=True)
    return pairwise_kernels(X, Y, metric='linear', n_jobs=n_jobs)


def compute_mmd(samples1, samples2, n_jobs=None):
    """
    MMD between two list of samples
    """
    # print('Compute_mmd', inspect.ArgSpec(compute_mmd))

    X = kernel_compute(samples1, n_jobs=n_jobs)
    Y = kernel_compute(samples2, n_jobs=n_jobs)
    Z = kernel_compute(samples1, Y=samples2, n_jobs=n_jobs)

    return np.average(X) + np.average(Y) - 2 * np.average(Z)


def smiles_to_nx(smiles):
    gs = []
    for smi in smiles:
        g = nx.Graph()
        mol = Chem.MolFromSmiles(smi)
        for a in mol.GetAtoms():
            g.add_node(a.GetIdx(), label=a.GetSymbol())
        for b in mol.GetBonds():
            g.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), label=b.GetBondType())
        gs.append(g)
    return gs


def chamfer_dist(old_smis, new_smis, num_procs=50):
    old_mols = [Chem.MolFromSmiles(smi) for smi in old_smis]
    new_mols = [Chem.MolFromSmiles(smi) for smi in new_smis]
    old_mols = list(filter(lambda x: x, old_mols))
    new_mols = list(filter(lambda x: x, new_mols))
    div = InternalDiversity()
    with mp.Pool(num_procs) as p:
        res = p.starmap(div.distance, tqdm(product(old_mols, new_mols), desc="chamfer", total=len(old_mols)*len(new_mols)))
    dists = np.array(res).reshape((len(old_mols), len(new_mols)))
    d1 = dists[dists.argmin(axis=0), list(range(dists.shape[1]))].mean()
    d2 = dists[list(range(dists.shape[0])), dists.argmin(axis=1)].mean()
    return (d1+d2)/2


def retro_sender(generated_samples, folder, retro_suffix=''):
    # File communication to obtain retro-synthesis rate
    sender_filename = os.path.join(folder, 'generated_samples.txt')
    receiver_filename = os.path.join(folder, 'output_syn.txt')
    open(sender_filename, 'w+').close() # clear
    open(receiver_filename, 'w+').close() # clear
    while(True):
        with open(sender_filename, 'r') as fr:
            editable = lock(fr)
            if editable:
                with open(sender_filename, 'w') as fw:
                    for sample in generated_samples:
                        fw.write('{}\n'.format(sample))                
                fcntl.flock(fr, fcntl.LOCK_UN)
                break
    num_samples = len(generated_samples)
    print("Waiting for retro_star evaluation...")
    count = 0
    latest = time.time()
    while(True):
        with open(receiver_filename, 'r') as fr:
            editable = lock(fr)
            if editable:
                syn_status = []
                lines = fr.readlines()
                if len(lines) == num_samples:
                    for idx, line in enumerate(lines):
                        splitted_line = line.strip().split()
                        syn_status.append((idx, splitted_line[2]))
                    break
                else:
                    if len(lines) > count:
                        count = len(lines)
                        latest = time.time()
                    elif time.time()-latest > 200:
                        for idx, line in enumerate(lines):
                            splitted_line = line.strip().split()
                            syn_status.append((idx, splitted_line[2]))
                        break                        
            fcntl.flock(fr, fcntl.LOCK_UN)
        time.sleep(1)
    generated_samples = [generated_samples[int(idx)] for idx, _ in syn_status]
    sender_filename_suffix = os.path.join(folder, f'generated_samples{retro_suffix}.txt')
    receiver_filename_suffix = os.path.join(folder, f'output_syn{retro_suffix}.txt')
    shutil.copyfile(sender_filename, sender_filename_suffix) # save res
    shutil.copyfile(receiver_filename, receiver_filename_suffix) # save res
    return np.mean([int(eval(s[1])) for s in syn_status])


def get_novelty(old_smiles, new_smiles):
    def canonicalize(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmarts(smi)        
        return Chem.MolToSmiles(mol)
    old_smiles = [canonicalize(smi) for smi in old_smiles]
    res = []
    for smi in new_smiles:  
        try:
            smi = canonicalize(smi)
        except:
            continue
        res.append(smi not in old_smiles)
    return sum(res)/len(res)


def get_uniqueness(smis):
    canon = set()
    uniq = 0
    for smi in smis:
        try:
            smi = Chem.CanonSmiles(smi)
        except:
            continue
        if smi not in canon:
            canon.add(smi)
            uniq += 1
    return uniq/len(smis)


def data_driven_membership(mols, old_dags, new_smiles):
    def dfs_count(dag, count):
        count[dag.val] = count.get(dag.val, 0) + 1
        for c in dag.children:
            if c[0].id == 0: # cyclic
                continue
            dfs_count(c[0], count)    
    # data driven way
    occur = {}
    all_counts = {}
    for dag in old_dags:
        count = {}
        dfs_count(dag, count)
        for k in count:
            occur[k] = occur.get(k, 0) + 1
            all_counts[k] = all_counts.get(k, 0) + count[k]
    freq = sorted(occur, key=lambda k: occur[k])
    common = freq[-1]
    assert 'G' in common        
    motif = mols[int(common.split('G')[-1])-1]    
    member = 0
    for smiles in new_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol.GetSubstructMatches(motif):
            member += 1
    return member/len(new_smiles)   


def check_hopv(smis):
    thio = Chem.MolFromSmiles('[cH:1]1[cH:2][cH:3][cH:4][s:5]1') # thiophene
    members = 0
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        members += bool(mol.GetSubstructMatches(thio))
    return members/len(smis)


def check_ptc(smis):    
    halide_smis = ['[CH2:1]([Br:2])[CH3:3]', '[Cl:1][CH3:2]']
    halides = [Chem.MolFromSmiles(s) for s in halide_smis]
    members = 0
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        members += max(*[bool(mol.GetSubstructMatches(halide)) for halide in halides])
    return members/len(smis)


def check_isocyanate(smis):
    # Define the SMARTS pattern for the Isocyanate group (-N=C=O)
    isocyanate_smarts = '[N]=C=O'    
    isocyanate_pattern = Chem.MolFromSmarts(isocyanate_smarts)
    members = 0
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        members += mol.HasSubstructMatch(isocyanate_pattern)    
    return members/len(smis)


def check_acrylate(smis):
    # Define the SMARTS pattern for the Isocyanate group (-N=C=O)
    acrylate_smarts = 'C=CC(=O)O'
    acrylate_pattern = Chem.MolFromSmarts(acrylate_smarts)
    members = 0
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        members += mol.HasSubstructMatch(acrylate_pattern)    
    return members/len(smis)


def check_chain_extender(smis):
    # Define SMARTS patterns for various chain extenders
    members = 0
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        members += mol.HasSubstructMatch(Chem.MolFromSmiles("CO")) or mol.HasSubstructMatch(Chem.MolFromSmiles("CN"))
    return members/len(smis)



def get_membership(args, new_smiles):    
    if 'hopv' in args.data_file.lower():
        func = check_hopv
    elif 'ptc' in args.data_file.lower():
        func = check_ptc
    elif 'isocyanates' in args.data_file.lower():
        func = check_isocyanate
    elif 'acrylate' in args.data_file.lower():
        func = check_acrylate
    elif 'chain_extender' in args.data_file.lower():
        func = check_chain_extender
    else:
        func = lambda smi: True
    return func(new_smiles)


def mol_from_smiles(smi):
    if smi:
        return Chem.MolFromSmiles(smi)
    else:
        return None



def compute_metrics(args, mols, old_smiles, new_smiles, retro_suffix='_test'):
    log_folder = args.log_folder if hasattr(args, 'log_folder') else args.logs_folder    
    metrics = {}       
    mols = [mol_from_smiles(smi) for smi in new_smiles]    
    metrics['valid'] = sum([mol is not None for mol in mols])/len(mols)
    mols = [mol for mol, smi in zip(mols, new_smiles) if mol is not None]
    new_smiles = [smi for smi in new_smiles if mol_from_smiles(smi) is not None]
    # following assumes new_smiles 100% validity
    metrics['membership'] = get_membership(args, new_smiles)
    if args.retro:
        if len(new_smiles) > 1000:
            random.shuffle(new_smiles)
            retro_smiles = new_smiles[:1000]
        else:
            retro_smiles = new_smiles    
    batch_size = 500
    n_jobs = 50
    device = 'cuda:0'
    print("compute ptest")
    ptest = compute_intermediate_statistics(old_smiles, n_jobs=n_jobs,
                                                device=device,
                                                batch_size=batch_size,
                                                pool=mp.Pool(50))
    print("finish ptest")
    kwargs = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    if args.moses:        
        metrics_all = get_all_metrics(new_smiles,n_jobs=n_jobs,test=old_smiles,train=old_smiles) 
        print("finish get_all_metrics")       
        metrics.update(metrics_all)
    else:
        fcd = FCDMetric(**kwargs)             
        metrics['FCD'] = fcd(gen=new_smiles, pref=ptest['FCD'])
        print("fcd metric done")  
        mols_for_scaf = list(filter(None, mapper(1)(get_mol, new_smiles)))
        scaf = ScafMetric(**kwargs) 
        metrics['Scaf'] = scaf(gen=mols_for_scaf, pref=ptest['Scaf'])    
        print("scaf metric done")  

    old_nx = smiles_to_nx(old_smiles)
    new_nx = smiles_to_nx(new_smiles)
    print("smiles to nx done") 
    metrics['NSPDK'] = compute_mmd(old_nx, new_nx, n_jobs=n_jobs)
    print("nspdk done") 

    if args.retro:
        metrics['RS'] = retro_sender(retro_smiles, log_folder, retro_suffix=retro_suffix)
    if old_smiles is not None:        
        metrics['novelty'] = get_novelty(old_smiles, new_smiles)
        metrics['chamfer'] = chamfer_dist(old_smiles, new_smiles) # not defined for invalid
    metrics['unique'] = get_uniqueness(new_smiles)
    div = InternalDiversity()    
    # random.shuffle(new_smiles)
    print(f"{len(mols)} valid")
    metrics['diversity'] = div.get_diversity(mols)    
    return metrics 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_folder")
    parser.add_argument("--old_smiles_file")
    parser.add_argument("--new_smiles_file")
    parser.add_argument("--retro", action='store_true')
    parser.add_argument("--moses", action='store_true')
    args = parser.parse_args()
    setattr(args, 'data_file', args.old_smiles_file)
    suffix = Path(args.new_smiles_file).stem
    old_smiles = [l.rstrip('\n') for l in open(args.old_smiles_file).readlines()]
    new_smiles = [l.rstrip('\n') for l in open(args.new_smiles_file).readlines()]
    metrics = compute_metrics(args, None, old_smiles, new_smiles)    
    json.dump(metrics, open(os.path.join(args.log_folder, f'metrics-for-{suffix}.json'), 'w+'))
    print(os.path.abspath(os.path.join(args.log_folder, f'metrics-for-{suffix}.json')))
    