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


import argparse
import os

import h5py
import numpy as np


def unpack(path):
    f = h5py.File(path, 'r')

    dat = {}
    for k in f.keys():
        dat[k] = f[k][()]

    f.close()

    return dat

def append(dat, other):
    frag_coord_off = dat['frag_data'].shape[0]
    frag_lig_smi_off = dat['frag_lig_smi'].shape[0]
    rec_coord_off = dat['rec_coords'].shape[0]

    # Update fragment coords.
    other['frag_lookup']['f1'] += frag_coord_off
    other['frag_lookup']['f2'] += frag_coord_off
    other['frag_lookup']['f3'] += frag_coord_off
    other['frag_lookup']['f4'] += frag_coord_off

    # Update receptor coords.
    other['rec_lookup']['f1'] += rec_coord_off
    other['rec_lookup']['f2'] += rec_coord_off

    # Update ligand index.
    other['frag_lig_idx'] += frag_lig_smi_off

    # Concatenate everything.
    for k in dat:
        dat[k] = np.concatenate((dat[k], other[k]), axis=0)

def cat_all(paths):
    dat = unpack(paths[0])

    for i in range(1, len(paths)):
        append(dat, unpack(paths[i]))

    return dat

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True, help='Path to folder containing intermediate .h5 fragments')
    parser.add_argument('-o', '--output', default='moad.h5', help='Output file path (.h5)')

    args = parser.parse_args()

    inp = args.input
    paths = [x for x in os.listdir(inp) if x.endswith('.h5')]
    paths = [os.path.join(inp, x) for x in paths]

    print('Merging:')
    for k in paths:
        print('- %s' % k)

    full = cat_all(paths)

    f = h5py.File(args.output, 'w')
    for k in dat:
        f[k] = dat[k]
    f.close()

    print('Done!')

if __name__=='__main__':
    main()
