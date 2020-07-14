
import os

import h5py

import numpy as np

import rdkit.Chem.AllChem as Chem
from rdkit.Chem.Descriptors import ExactMolWt

import torch
import torch.nn as nn
import torch.nn.functional as F

from leadopt.models.voxel import VoxelFingerprintNet2, VoxelFingerprintNet2b


class SavedModel(object):
    
    model_class = None
    model_args = None
    
    @classmethod
    def load(cls, path):
        m = cls.model_class(**cls.model_args).cuda()
        m.load_state_dict(torch.load(path))
        m.eval()
        return m
    
    @classmethod
    def get_fingerprints(cls, path, max_weight=None):
        f = h5py.File(os.path.join(path, cls.fingerprint_data), 'r')
        
        data = f['fingerprints'][()]
        smiles = f['smiles'][()]
        
        f.close()

        if max_weight is not None:
            # compute weight for each fingerprint
            sm_weight = np.zeros(len(smiles))
            for i in range(len(smiles)):
                mol = Chem.MolFromSmiles(smiles[i].decode('ascii'), sanitize=False)
                mol.UpdatePropertyCache(strict=False)
                sm_weight[i] = ExactMolWt(mol)
            
            # apply bounds
            valid_sm = sm_weight <= max_weight

            data = data[valid_sm]
            smiles = smiles[valid_sm]
        
        return data, smiles
    
class V2_RDK_M150(SavedModel):
    model_class = VoxelFingerprintNet2
    model_args = {
        'in_channels': 18,
        'output_size': 2048,
        'batchnorm': True,
        'sigmoid': True
    }
    
    grid_width=24
    grid_res=1
    receptor_types=[6,7,8,9,15,16,17,35,53]
    parent_types=[6,7,8,9,15,16,17,35,53]
    
    fingerprint_data = 'fingerprint_rdk_2048.h5'


class V2b_RDK_M150(SavedModel):
    model_class = VoxelFingerprintNet2b
    model_args = {
        'in_channels': 18,
        'output_size': 2048,
        'pad': False,
        'blocks': [32, 64],
        'fc': [2048]
    }
    
    grid_width=24
    grid_res=1
    receptor_types=[6,7,8,9,15,16,17,35,53]
    parent_types=[6,7,8,9,15,16,17,35,53]
    
    fingerprint_data = 'fp_rdk.h5'


MODELS = {
    'rdk_m150': V2_RDK_M150,
    'v2b_rdk_m150': V2b_RDK_M150
}
