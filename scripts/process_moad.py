'''
Utility script to convert the MOAD dataset into a packed format
'''
import sys
import argparse
import os
import re
import multiprocessing
import threading

from moad_util import parse_moad

import rdkit.Chem.AllChem as Chem
from rdkit.Chem.Descriptors import ExactMolWt
import molvs
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
# - frag_lig_id:    [lig_id]
# - frag_coords:    [x][y][z]
# - frag_types:     [num]
# - frag_smiles:    [smiles]
# - frag_mass:      [mass]
# - frag_dist:      [dist]

LOAD_TIMEOUT = 60


u = molvs.charge.Uncharger()
t = molvs.tautomer.TautomerCanonicalizer()


REPLACE = [
    ('C(=O)[O-]', 'C(=O)O'),
    ('N=[N+]=N', 'N=[N+]=[N-]'),
    ('[N+](=O)O', '[N+](=O)[O-]'),
    ('S(O)(O)O', '[S+2](O)(O)O'),
]


def basic_replace(sm):
    m = Chem.MolFromSmiles(sm, False)
    
    for a,b in REPLACE:
        m = Chem.ReplaceSubstructs(
            m,
            Chem.MolFromSmiles(a, sanitize=False),
            Chem.MolFromSmiles(b, sanitize=False),
            replaceAll=True
        )[0]
        
    return Chem.MolToSmiles(m)


def neutralize_smiles(sm):
    m = Chem.MolFromSmiles(sm)
    m = u.uncharge(m)
    m = t.canonicalize(m)
    sm = Chem.MolToSmiles(m)
    sm = basic_replace(sm)
    
    try:
        return molvs.standardize_smiles(sm)
    except:
        print(sm)
        return sm


def load_example(base, rec_id, target):

    rec_path = os.path.join(base, '%s_rec.pdb' % rec_id)

    # Load receptor data.
    rec_coords, rec_types = util.load_receptor_ob(rec_path)

    # (frag_data, parent_data, smiles, mass, dist, lig_off)
    fragments = []

    # (smi)
    lig_smiles = []


    lig_off = 0
    for lig in target.ligands:
        
        lig_path = os.path.join(base, '%s_%s.pdb' % (rec_id, lig[0].replace(' ','_')))
        try:
            lig_mol = Chem.MolFromPDBFile(lig_path, True)
        except:
            continue

        if lig_mol is None:
            continue

        lig_smi = lig[1]
        lig_smiles.append(lig_smi)

        ref = Chem.MolFromSmiles(lig_smi)
        lig_fixed = Chem.AssignBondOrdersFromTemplate(ref, lig_mol)

        splits = util.generate_fragments(lig_fixed)

        for parent, frag in splits:
            frag_data = util.mol_array(frag)
            parent_data = util.mol_array(parent)
            
            frag_smi = Chem.MolToSmiles(
                frag, 
                isomericSmiles=False, 
                kekuleSmiles=False, 
                canonical=True, 
                allHsExplicit=False
            )

            frag_smi = neutralize_smiles(frag_smi)
            
            frag.UpdatePropertyCache(strict=False)
            mass = ExactMolWt(frag)
            
            dist = util.frag_dist_to_receptor_raw(rec_coords, frag)
            
            fragments.append((frag_data, parent_data, frag_smi, mass, dist, lig_off))

        lig_off += 1

    return (rec_coords, rec_types, fragments, lig_smiles)

def do_thread(out, args):
    try:
        out[0] = load_example(*args)
    except:
        out[0] = None

def multi_load(packed):
    out = [None]

    t = threading.Thread(target=do_thread, args=(out, packed))
    t.start()
    t.join(timeout=LOAD_TIMEOUT)

    if t.is_alive():
        print('timeout', packed[1])

    return (packed, out[0])


def process(work, processed, moad_csv, out_path='moad.h5', num_cores=1):
    '''Process MOAD data and save to a packed format.

    Args:
    - out_path: where to save the .h5 packed data
    '''
    rec_lookup = [] # (id, start, end)
    rec_coords = [] # (x,y,z)
    rec_types = [] # (num, aro, hdon, hacc, pcharge)

    frag_lookup = [] # (id, f_start, f_end, p_start, p_end)
    frag_lig_idx = [] # (lig_idx)
    frag_lig_smi = [] # (lig_smi)
    frag_data = [] # (x,y,z,type)
    frag_smiles = [] # (frag_smi)
    frag_mass = [] # (mass)
    frag_dist = [] # (dist)

    # Data pointers.
    rec_i = 0
    frag_i = 0

    # Multiprocess.
    with multiprocessing.Pool(num_cores) as p:
        with tqdm.tqdm(total=len(work)) as pbar: 
            for w, res in p.imap_unordered(multi_load, work):
                pbar.update()

                if res == None:
                    print('[!] Failed: %s' % w[1])
                    continue
        
                rcoords, rtypes, fragments, ex_lig_smiles = res

                if len(fragments) == 0:
                    print('Empty', w[1])
                    continue

                rec_id = w[1]

                # Add receptor info.
                rec_start = rec_i
                rec_end = rec_i + rcoords.shape[0]
                rec_i += rcoords.shape[0]

                rec_coords.append(rcoords)
                rec_types.append(rtypes)
                rec_lookup.append((rec_id.encode('ascii'), rec_start, rec_end))

                lig_idx = len(frag_lig_smi)

                # Add fragment info.
                for fdat, pdat, frag_smi, mass, dist, lig_off in fragments:
                    frag_start = frag_i
                    frag_end = frag_i + fdat.shape[0]
                    frag_i += fdat.shape[0]
                    
                    parent_start = frag_i
                    parent_end = frag_i + pdat.shape[0]
                    frag_i += pdat.shape[0]
                    
                    frag_data.append(fdat)
                    frag_data.append(pdat)
                    
                    frag_lookup.append((rec_id.encode('ascii'), frag_start, frag_end, parent_start, parent_end))
                    frag_lig_idx.append(lig_idx+lig_off)
                    frag_smiles.append(frag_smi)
                    frag_mass.append(mass)
                    frag_dist.append(dist)

                # Add ligand smiles.
                frag_lig_smi += ex_lig_smiles
            
    # Convert to numpy format.
    print('Convert numpy...', flush=True)
    n_rec_lookup = np.array(rec_lookup, dtype='<S16,<i4,<i4')
    n_rec_coords = np.concatenate(rec_coords, axis=0).astype(np.float32)
    n_rec_types = np.concatenate(rec_types, axis=0).astype(np.int32)

    n_frag_data = np.concatenate(frag_data, axis=0)
    n_frag_lookup = np.array(frag_lookup, dtype='<S16,<i4,<i4,<i4,<i4')
    n_frag_lig_idx = np.array(frag_lig_idx)
    n_frag_lig_smi = np.array(frag_lig_smi, dtype='<S')
    n_frag_smiles = np.array(frag_smiles, dtype='<S')
    n_frag_mass = np.array(frag_mass)
    n_frag_dist = np.array(frag_dist)

    # Save.
    with h5py.File(os.path.join(out_path), 'w') as f:
        f['rec_lookup'] = n_rec_lookup
        f['rec_coords'] = n_rec_coords
        f['rec_types'] = n_rec_types

        f['frag_data'] = n_frag_data
        f['frag_lookup'] = n_frag_lookup
        f['frag_lig_idx'] = n_frag_lig_idx
        f['frag_lig_smi'] = n_frag_lig_smi
        f['frag_smiles'] = n_frag_smiles
        f['frag_mass'] = n_frag_mass
        f['frag_dist'] = n_frag_dist

    print('Done!')


def build_multiple(processed, moad_csv, out_path='moad.h5', num_cores=1, size=5000):
    # Load MOAD metadata.
    moad_families, moad_targets = parse_moad(moad_csv)
    
    # Load files.
    files = [x for x in os.listdir(processed) if not x.startswith('.')]
    rec = [x for x in files if '_rec' in x]

    print('[*] Found %d receptors' % len(rec))

    # Create work queue.
    work = []
    for rec_name in sorted(rec):
        rec_id = rec_name.split('_rec')[0]
        pdb_id = rec_id.split('_')[0].upper()
        if pdb_id not in moad_targets:
            print('missing pdb', pdb_id)
            continue
        target = moad_targets[pdb_id]

        work.append((processed, rec_id, target))

    split = 0
    for i in range(split * size, len(work), size):
        print('Building split %d' % split)
        out_name = out_path.replace('.h5', '_%d.h5' % split)
        split += 1

        process(work[i:i+size], processed, moad_csv, out_name, num_cores)

    print('Done all!')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--datasets', required=True, help='Path to MOAD /processed folder')
    parser.add_argument('-c', '--csv', required=True, help='Path to MOAD "every.csv"')
    parser.add_argument('-o', '--output', default='fragments.h5', help='Output file path (.h5)')
    parser.add_argument('-n', '--num_cores', default=1, type=int, help='Number of cores')
    parser.add_argument('-s', '--size', default=500, type=int, help='Number of targets per output dataset')

    args = parser.parse_args()

    build_multiple(args.datasets, args.csv, args.output, args.num_cores, args.size)


if __name__=='__main__':
    main()
