'''
Utility script to convert the pdbbind dataset into a packed format
'''
import sys
import argparse
import os
import re

import rdkit.Chem.AllChem as Chem
from rdkit.Chem.Descriptors import ExactMolWt

import h5py
import tqdm
import numpy as np

# add leadopt to path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from leadopt import util



# Data Format
# 
# Receptor data:
# - rec_lookup:     [id][start][end]
# - rec_coords:     [x][y][z]
# - rec_types:      [num][is_hacc][is_hdon][is_aro][pcharge]
#
# Fragment data:
# - frag_lookup:    [id][fstart][fend][pstart][pend]
# - frag_coords:    [x][y][z]
# - frag_types:     [num]
# - frag_smiles:    [smiles]
# - frag_mass:      [mass]
# - frag_dist:      [dist]


def load_example(path, prot_id):
    '''
    load a single protein/ligand pair example
    '''
    rec_path = os.path.join(path, '%s_protein.pdb' % prot_id)
    lig_path = os.path.join(path, '%s_ligand.sdf' % prot_id)
    
    # load fragments
    lig, frags = util.load_ligand(lig_path)

    # load receptor data
    rec_coords, rec_types = util.load_receptor_ob(rec_path)
    
    fragments = [] # (frag_data, parent_data, smiles, mass, dist)
    for parent, frag in frags:
        frag_data = util.mol_array(frag)
        parent_data = util.mol_array(parent)
        
        smiles = Chem.MolToSmiles(Chem.RemoveHs(frag, sanitize=False), isomericSmiles=False)
        # smiles = re.sub(r'\[\d*\*\]', '*', smiles) # remove extra information about dummy atom
        
        frag.UpdatePropertyCache(strict=False)
        mass = ExactMolWt(frag)
        
        dist = util.frag_dist_to_receptor_raw(rec_coords, frag)
        
        fragments.append((frag_data, parent_data, smiles, mass, dist))
        
    return rec_coords, rec_types, fragments


def process(datasets, out_path='pdbbind.h5'):
    '''
    process pdbbind data and save to a packed format

    * datasets: a list of folders containing data structured like:
        - aaaa
            - aaaa_protein.pdb
            - aaaa_ligand.sdf
        ...
    * out_path: where to save the .h5 packed data
    '''

    rec_lookup = [] # (id, start, end)
    rec_coords = [] # (x,y,z)
    rec_types = [] # (num, aro, hdon, hacc, pcharge)

    frag_lookup = [] # (id, f_start, f_end, p_start, p_end)
    frag_data = [] # (x,y,z,type)
    frag_smiles = [] # (smiles)
    frag_mass = [] # (mass)
    frag_dist = [] # (dist)

    rec_i = 0
    frag_i = 0

    for path in datasets:
        for prot_id in tqdm.tqdm(sorted(os.listdir(path))):
            if prot_id.startswith('.') or prot_id in ['index', 'readme']:
                continue
            
            example_path = os.path.join(path,prot_id)
            rcoords, rtypes, fragments = load_example(example_path, prot_id)

            # add receptor info
            rec_start = rec_i
            rec_end = rec_i + rcoords.shape[0]
            rec_i += rcoords.shape[0]
            
            rec_coords.append(rcoords)
            rec_types.append(rtypes)
            rec_lookup.append((prot_id, rec_start, rec_end))
            
            # add fragment info
            for fdat, pdat, smiles, mass, dist in fragments:
                frag_start = frag_i
                frag_end = frag_i + fdat.shape[0]
                frag_i += fdat.shape[0]
                
                parent_start = frag_i
                parent_end = frag_i + pdat.shape[0]
                frag_i += pdat.shape[0]
                
                frag_data.append(fdat)
                frag_data.append(pdat)
                
                frag_lookup.append((prot_id, frag_start, frag_end, parent_start, parent_end))
                frag_smiles.append(smiles)
                frag_mass.append(mass)
                frag_dist.append(dist)
            
    # convert to numpy format
    n_rec_lookup = np.array(rec_lookup, dtype='<S4,<i4,<i4')
    n_rec_coords = np.concatenate(rec_coords, axis=0).astype(np.float32)
    n_rec_types = np.concatenate(rec_types, axis=0).astype(np.int32)

    n_frag_data = np.concatenate(frag_data, axis=0)
    n_frag_lookup = np.array(frag_lookup, dtype='<S4,<i4,<i4,<i4,<i4')
    n_frag_smiles = np.array(frag_smiles, dtype='<S')
    n_frag_mass = np.array(frag_mass)
    n_frag_dist = np.array(frag_dist)

    # save
    with h5py.File(os.path.join(out_path), 'w') as f:
        f['rec_lookup'] = n_rec_lookup
        f['rec_coords'] = n_rec_coords
        f['rec_types'] = n_rec_types

        f['frag_data'] = n_frag_data
        f['frag_lookup'] = n_frag_lookup
        f['frag_smiles'] = n_frag_smiles
        f['frag_mass'] = n_frag_mass
        f['frag_dist'] = n_frag_dist

    print('Done!')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--datasets', required=True, nargs='+', help='List of dataset folders to include')
    parser.add_argument('-o', '--output', default='fragments.h5', help='Output file path (.h5)')

    args = parser.parse_args()

    process(args.datasets, out_path=args.output)


if __name__=='__main__':
    main()