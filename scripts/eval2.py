
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from leadopt import data_util
from config import partitions
from leadopt.models.voxel import VoxelFingerprintNet2b
from leadopt.grid_util import get_batch
from leadopt.model_conf import top_k_acc_mse

import numpy as np

import tqdm

# FRAG_FILE = '/home/hag63/DataB/hag63/data/pdbbind_v2/fragments_desc2.h5'
# FP_FILE = '/home/hag63/DataB/hag63/data/pdbbind_v2/fp_rdk_desc.h5'

# MASS_MAX = 150
# TYP = 'simple_h'

# MODEL_FILE = '/home/hag63/DataB/hag63/models/200618_m150_simple_h.pt'
# IN_CHANNELS = 9
# REC_CHANNELS = 5
# PARENT_CHANNELS = 4

# IGNORE_REC = False
# IGNORE_PARENT = False

def mse(yp, yt):
    return torch.sum((yp - yt) ** 2, axis=1)

def bce(yp, yt):
    return torch.sum(F.binary_cross_entropy(yp,yt,reduction='none'), axis=1)

def tanimoto(yp, yt):
    intersect = torch.sum(yt * torch.round(yp), axis=1)
    union = torch.sum(torch.clamp(yt + torch.round(yp), 0, 1), axis=1)
    return 1 - (intersect / union)

_cos = nn.CosineSimilarity(dim=1, eps=1e-6)
def cos(yp, yt):
    return 1 - _cos(yp,yt)


DIST_METRIC = {
    'mse': mse,
    'bce': bce,
    'tanimoto': tanimoto,
    'cos': cos
}


def run():
    
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('-f', '--fragments', required=True)
    parser.add_argument('-fp', '--fingerprints', required=True)

    # grid generration
    # parser.add_argument('--grid_width',type=int,default=24)
    # parser.add_argument('--grid_res',type=float,default=1)

    # fragment filtering
    # parser.add_argument('--fdist_min',type=float)
    # parser.add_argument('--fdist_max',type=float)
    # parser.add_argument('--fmass_min',type=float)
    parser.add_argument('--fmass_max',type=float)

    # receptor/parent options
    parser.add_argument('--ignore_receptor',action='store_true',default=False)
    parser.add_argument('--ignore_parent',action='store_true',default=False)
    parser.add_argument('-rec_typer', required=True, choices=[k for k in data_util.REC_TYPER])
    parser.add_argument('-lig_typer', required=True, choices=[k for k in data_util.LIG_TYPER])
    parser.add_argument('-rec_channels', required=True, type=int)
    parser.add_argument('-lig_channels', required=True, type=int)

    # dist
    parser.add_argument('-dist',default='mse')

    # model file
    parser.add_argument('-model')

    # parse arguments
    args = parser.parse_args()

    print(args)

    do_eval(args)


def mse(p, fp):
    return torch.sum((fp-p)**2, axis=1)

def evaluate(args, model, data, fpdat, idx):
    t, fp, _, _ = get_batch(
        data,
        rec_channels=args.rec_channels,
        parent_channels=args.lig_channels,
        batch_size=1, 
        width=24, 
        res=1, 
        ignore_receptor=args.ignore_receptor,
        ignore_parent=args.ignore_parent,
        include_freq=True,
        batch_set=[idx]
    )

    dist_fn = DIST_METRIC[args.dist]
    
    # true index
    fp_idx = int(torch.where(mse(fp.cpu(), fpdat)==0)[0][0].numpy())

    with torch.no_grad():
        p = model(t).cpu()
        
    dist = dist_fn(p, fpdat).numpy()
    
    return fp_idx, dist

def eval_batch(args, model, data, fpdat, pre=''):

    idx = []
    dist = []

    for i in tqdm.tqdm(range(len(data))):
        fp_i, d = evaluate(args, model, data, fpdat, i)

        idx.append(fp_i)
        dist.append(d)

    np.save('./%s_idx.npy' % pre, np.array(idx))
    np.save('./%s_dist.npy' % pre, np.array(dist))


def do_eval(args):

    # load data
    test_data = data_util.FragmentDataset(
        args.fragments,
        args.fingerprints,
        rec_typer=data_util.REC_TYPER[args.rec_typer],
        lig_typer=data_util.LIG_TYPER[args.lig_typer],
        filter_rec=partitions.TEST,
        fdist_min=0.5,
        fdist_max=3,
        fmass_min=None,
        fmass_max=args.fmass_max,
        verbose=True
    )

    fp = test_data.fingerprints
    data = fp['fingerprint_data']
    smi = fp['fingerprint_smiles']
    all_fp = torch.Tensor(data)
    valid_fp = torch.Tensor(data[test_data.valid_fingerprints])

    # np.save('./fpdat.npy', data)
    np.save('./fpsmi.npy', smi)
    np.save('./valid_fp.npy', test_data.valid_fingerprints)

    # load model
    m = VoxelFingerprintNet2b(
        in_channels=args.rec_channels + args.lig_channels,
        output_size=2048,
        blocks=[32,64],
        fc=[2048],
        pad=False
    )
    m.load_state_dict(torch.load(args.model))
    m = m.cuda()
    m = m.eval()

    eval_batch(args, m, test_data, all_fp, 'all')
    eval_batch(args, m, test_data, valid_fp, 'valid')

    print('done.')


if __name__=='__main__':
    run()
