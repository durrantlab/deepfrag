'''
fragment prediction CLI tool
'''

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import h5py

from leadopt.models.voxel import VoxelFingerprintNet2
from leadopt.infer import infer_all


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
    def get_fingerprints(cls, path):
        f = h5py.File(os.path.join(path, cls.fingerprint_data), 'r')
        
        data = f['fingerprints'][()]
        smiles = f['smiles'][()]
        
        f.close()
        
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


MODELS = {
    'rdk_m150': V2_RDK_M150
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--receptor', required=True, help='Receptor file (.pdb)')
    parser.add_argument('-l', '--ligand', required=True, help='Ligand file (.sdf)')
    
    parser.add_argument('-n', '--num_samples', type=int, default=16, help='Number of random rotation samples to use per prediction (default: 16)')
    parser.add_argument('-k', '--num_suggestions', type=int, default=25, help='Number of suggestions per fragment')

    parser.add_argument('-m', '--model', default=[k for k in MODELS][0], choices=[k for k in MODELS])
    parser.add_argument('-mp', '--model_path', required=True)
    parser.add_argument('-d', '--data_path', required=True)

    args = parser.parse_args()

    # load model
    m = MODELS[args.model].load(args.model_path)
    fingerprints, smiles = MODELS[args.model].get_fingerprints(args.data_path)

    # run infer step
    res = infer_all(
        model=m, 
        fingerprints=fingerprints, 
        smiles=smiles, 
        rec_path=args.receptor, 
        lig_path=args.ligand, 
        num_samples=args.num_samples,
        k=args.num_suggestions
    )

    print(res)


if __name__=='__main__':
    main()


# leadopt.py -r my_receptor.pdb -l my_ligand.sdf