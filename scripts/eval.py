'''
evaluate a trained model and collect some stats
'''


import sys
import argparse
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import h5py
import tqdm
import numpy as np

# add leadopt to path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from leadopt import util
from leadopt.models.voxel import VoxelFingerprintNet2b
from leadopt.data_util import FragmentDataset
from leadopt.grid_util import get_batch_dual, get_batch
from config import partitions


def load_model(model):
    print('[*] Load model...')
    m = VoxelFingerprintNet2b(in_channels=18, output_size=2048, blocks=[32,64], fc=[2048], pad=False)
    m.load_state_dict(torch.load(model))
    m.eval()
    m = m.cuda()

    return m


def load_data(fragments, fingerprints, fmass_max):
    print('[*] Loading data...')
    train_dat = FragmentDataset(
        fragments,
        fingerprints, 
        partitions.TRAIN,
        fdist_min=0.5,
        fdist_max=3,
        fmass_max=fmass_max
    )

    test_dat = FragmentDataset(
        fragments,
        fingerprints, 
        partitions.TEST,
        fdist_min=0.5,
        fdist_max=3,
        fmass_max=fmass_max
    )

    return train_dat, test_dat

def mse(a,b):
    return np.sum((a-b)**2, axis=1)

def lookup(fp, fingerprints):
    '''lookup smiles by fragment'''
    fp_idx = np.argmin(mse(fingerprints, fp))
    return fp_idx

def get_pos(p, target, fingerprints):
    # target distance
    tdist = mse(p, target)
    
    # all other distances
    fdist = mse(fingerprints, p)
    
    # how many are closer?
    pos = np.sum(fdist < tdist)
    
    return pos

def get_pred(model, dat, fingerprints, idx, samples=1):
    t,fp,_ = get_batch(
        dat,
        batch_set=[idx] * samples,
        batch_size=samples,
        width=24,
        res=1
    )

    p = model(t).cpu().detach().numpy()
    fp = fp.cpu().numpy()
    
    fp = np.mean(fp, axis=0).reshape(1,-1)
    p = np.mean(p, axis=0).reshape(1,-1)
    
    return p, fp, lookup(fp, fingerprints)

def compute_pos_all(model, fingerprints, dat, samples=1):
    pos = np.zeros(len(dat))
    
    for i in tqdm.tqdm(range(len(dat))):
        p,fp,idx = get_pred(model, dat, fingerprints, i, samples)
        pos[i] = get_pos(p,fp,fingerprints)
        
    return pos

def process(args):
    
    m = load_model(args.model)
    train_dat, test_dat = load_data(args.fragments, args.fingerprints, args.fmass_max)

    # collect valid fingerprints
    fp_set = sorted(list(set(train_dat.valid_fingerprints) | set(test_dat.valid_fingerprints)))
    fingerprints = train_dat.fingerprints['fingerprint_data'][fp_set]
    smiles = train_dat.fingerprints['fingerprint_smiles'][fp_set]

    N = 3
    SAMPLES = [1,2,4,8,16,32]

    for k in SAMPLES:
        for i in range(N):
            print('%d :: %d' % (k,i))
            pos = compute_pos_all(m, fingerprints, test_dat, samples=k)
            np.save('./pos_%d_n%d' % (k, i), pos)

    print('done!')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--fragments', required=True, help='Path to fragemnts.h5 containing "rec_data" array')
    parser.add_argument('-fp', '--fingerprints', required=True, help='Path to fragemnts.h5 containing "rec_data" array')
    parser.add_argument('-m', '--model', required=True, help='Path to fragemnts.h5 containing "rec_data" array')
    parser.add_argument('--fmass_max', required=True, type=int)

    args = parser.parse_args()

    process(args)


if __name__=='__main__':
    main()
