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


import prody
import tqdm
import numpy as np
import argparse
import os
import multiprocessing

from moad_util import parse_moad

prody.utilities.logger.LOGGER.verbosity = 'none'


def load_example(path, target):
    m = prody.parsePDB(path)

    rec = m.select('not (nucleic or hetatm) and not water')

    # (lig_name, atoms)
    ligands = []
    for lig in target.ligands:
        lig_name, lig_chain, lig_resnum = lig[0].split(':')

        sel = None
        if len(lig_name) > 3 and lig_resnum == '1':
            # assume peptide, take the whole chain
            sel = m.select('chain %s' % lig_chain)
        else:
            # assume small molecule, single residue
            sel = m.select('chain %s and resnum = %s' % (lig_chain, lig_resnum))

        ligands.append((lig[0], sel))

    return rec, ligands


def do_proc(packed):
    out_dir, rec_name, path, target = packed

    try:
        rec, ligands = load_example(path, target)

        # Save receptor.
        prody.writePDB(os.path.join(out_dir, rec_name + '_rec.pdb'), rec)

        # Save ligands.
        for lig_name, lig_sel in ligands:
            if lig_sel is None:
                continue
            lig_name = lig_name.replace(' ', '_')
            prody.writePDB(os.path.join(out_dir, rec_name + '_' + lig_name + '.pdb'), lig_sel)

    except Exception as e:
        print('failed', path)
        print(e)

    return None


def load_all(moad_dir, moad_csv, out_dir, num_cores=1):
    computed = []

    if os.path.exists(out_dir):
        names = os.listdir(out_dir)
        names = [x.split('_rec')[0] for x in names if '_rec' in x]
        computed = names
    else:
        os.mkdir(out_dir)

    # Load MOAD csv.
    moad_families, moad_targets = parse_moad(moad_csv)

    # Collect input files.
    files = []
    for fname in os.listdir(moad_dir):
        if fname.startswith('.'):
            continue
        files.append(os.path.join(moad_dir, fname))

    files = sorted(files)
    print('[*] Loading %d files...' % len(files))

    failed = []
    info = {}

    # (path, target)
    work = []

    for path in tqdm.tqdm(files):
        rec_name = os.path.split(path)[-1].replace('.','_')
        if rec_name in computed:
            # Skip computed.
            continue

        pdb_id = rec_name.split('_')[0].upper()
        target = moad_targets[pdb_id]

        work.append((out_dir, rec_name, path, target))

    print('[*] Starting...')
    with multiprocessing.Pool(num_cores) as p:
        with tqdm.tqdm(total=len(work)) as pbar:
            for r in p.imap_unordered(do_proc, work):
                pbar.update()

    print('[*] Done.')


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', required=True, help='Path to MOAD folder')
    parser.add_argument('-c', '--csv', required=True, help='Path to MOAD "every.csv"')
    parser.add_argument('-o', '--output', default='./processed', help='Output directory')
    parser.add_argument('-n', '--num_cores', default=1, type=int, help='Number of cores')

    args = parser.parse_args()

    load_all(args.dataset, args.csv, args.output, args.num_cores)


if __name__=='__main__':
    run()
