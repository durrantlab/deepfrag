
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import rdkit.Chem.AllChem as Chem

import numpy as np
import h5py
import tqdm

from leadopt.grid_util import get_raw_batch
from leadopt.util import generate_fragments


def get_nearest_fp(fingerprints, fp, k=10):
    '''
    Return the top-k closest rows in fingerprints

    Returns [(idx1, dist1), (idx2, dist2), ...]
    '''
    def mse(a,b):
        return np.sum((a-b)**2, axis=1)

    d = mse(fingerprints, fp)
    arr = [(i,d[i]) for i in range(len(d))]
    arr = sorted(arr, key=lambda x: x[1])
    
    return arr[:k]


def infer_all(model, fingerprints, smiles, rec_path, lig_path, num_samples=16, k=25):
    '''

    '''
    # load ligand and receptor
    lig, frags = load_ligand(lig_path)
    rec = load_receptor(rec_path)

    # compute shared receptor coords and layers
    rec_coords, rec_layers = mol_to_points(rec)

    # [
    #   (parent_sm, orig_frag_sm, conn, [
    #       (new_frag_sm, merged_sm, score),
    #       ...
    #   ])
    # ]
    res = []

    for parent, frag in frags:
        # compute parent coords and layers
        parent_coords, parent_layers = mol_to_points(parent)

        # find connection point
        conn = get_connection_point(frag)

        # generate batch
        grid = get_raw_batch(rec_coords, rec_layers, parent_coords, parent_layers, conn)

        # infer
        fp = model(grid).detach().cpu().numpy()
        fp_mean = np.mean(fp, axis=0)

        # find closest fingerprints
        top = get_nearest_fp(fingerprints, fp_mean, k=k)

        # convert to (frag_smiles, merged_smiles, score) tuples
        top_smiles = [(smiles[x[0]], merge_smiles(Chem.MolToSmiles(parent), smiles[x[0]]), x[1]) for x in top]

        res.append(
            (Chem.MolToSmiles(parent), Chem.MolToSmiles(frag), tuple(conn), top_smiles)
        )

    return res
