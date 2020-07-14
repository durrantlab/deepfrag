
import sys
import argparse
import os
import re

import rdkit.Chem.AllChem as Chem
from rdkit.Chem.Descriptors import ExactMolWt

import matplotlib.pyplot as plt

import h5py
import tqdm
import numpy as np

# add leadopt to path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from leadopt import util


def make_graph(atom_types):
    pairs = [(k,atom_types[k]) for k in atom_types]
    spairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(12,8))
    plt.bar(range(len(spairs)), [x[1] for x in spairs], log=True)
    plt.xticks(range(len(spairs)), [int(x[0]) for x in spairs])
    plt.title('PDBBind Receptor Atom Count')
    plt.xlabel('Atomic number')
    plt.ylabel('Count')


def process(fragments):
    # load fragments
    dat = h5py.File(fragments, 'r')
    rec_data = dat['rec_data'][()]
    dat.close()

    # aggregate by atomic number
    atom_types = {}
    for i in range(len(rec_data)):
        t = rec_data[i][3]
        
        if not t in atom_types:
            atom_types[t] = 0
            
        atom_types[t] += 1

    print(atom_types)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--fragments', required=True, help='Path to fragemnts.h5 containing "rec_data" array')

    args = parser.parse_args()

    process(args.fragments)


if __name__=='__main__':
    main()
