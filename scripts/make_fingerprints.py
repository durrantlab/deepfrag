# Copyright 2021 Jacob Durrant

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.


'''
Utility script to generate fingerprints for smiles strings
'''
import argparse

import h5py
import numpy as np
import tqdm

from rdkit.Chem import rdMolDescriptors
import rdkit.Chem.AllChem as Chem
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D


def fold_to(bv, size=2048):
    '''fold a SparseBitVec to a certain length'''
    fp = np.zeros(size)

    for b in list(bv.GetOnBits()):
        fp[b % size] = 1

    return fp


def rdkfingerprint(m):
    '''rdkfingerprint as 2048-len bit array'''
    fp = Chem.rdmolops.RDKFingerprint(m)
    n_fp = list(map(int, list(fp.ToBitString())))
    return n_fp


def rdkfingerprint10(m):
    '''rdkfingerprint as 2048-len bit array (maxPath=10)'''
    fp = Chem.rdmolops.RDKFingerprint(m, maxPath=10)
    n_fp = list(map(int, list(fp.ToBitString())))
    return n_fp


def morganfingerprint(m):
    '''morgan fingerprint as 2048-len bit array'''
    m.UpdatePropertyCache(strict=False)
    Chem.rdmolops.FastFindRings(m)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2)
    n_fp = list(map(int, list(fp.ToBitString())))
    return n_fp


def gobbi2d(m):
    '''gobbi 2d pharmacophore as 2048-len bit array'''
    m.UpdatePropertyCache(strict=False)
    Chem.rdmolops.FastFindRings(m)
    bv = Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory)
    n_fp = fold_to(bv, size=2048)
    return n_fp


FINGERPRINTS = {
    'rdk': (rdkfingerprint, 2048),
    'rdk10': (rdkfingerprint10, 2048),
    'morgan': (morganfingerprint, 2048),
    'gobbi2d': (gobbi2d, 2048),
}


def process(fragments_path, fp_func, fp_size, out_path):
    # open fragments file
    f = h5py.File(fragments_path, 'r')
    smiles = f['frag_smiles'][()]
    f.close()

    # deduplicate smiles strings
    all_smiles = list(set(smiles))
    n_smiles = np.array(all_smiles)

    n_fingerprints = np.zeros((len(all_smiles), fp_size))

    for i in tqdm.tqdm(range(len(all_smiles))):
        # generate fingerprint
        m = Chem.MolFromSmiles(all_smiles[i].decode('ascii'), sanitize=False)
        n_fingerprints[i] = fp_func(m)

    # save
    with h5py.File(out_path, 'w') as f:
        f['fingerprints'] = n_fingerprints
        f['smiles'] = n_smiles

    print('Done!')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--fragments', required=True, help='Path to fragemnts.h5 containing "frag_smiles" array')
    parser.add_argument('-fp', '--fingerprint', required=True, choices=[k for k in FINGERPRINTS], help='Which fingerprint type to generate')
    parser.add_argument('-o', '--output', default='fingerprints.h5', help='Output file path (.h5)')

    args = parser.parse_args()

    fn, size = FINGERPRINTS[args.fingerprint]
    process(args.fragments, fn, size, out_path=args.output)


if __name__=='__main__':
    main()
