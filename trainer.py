
import os

import pybel
import tqdm
import numpy as np

import mol
import grid

from keras.utils import to_categorical

# atom types to consider
ATOM_TYPES = [
    6, 7, 8, 9, 15, 16, 17, 35, 53
]

# mapping
# atomic number -> van der waals radius
VDW_RADIUS = {
    6: 1.9,
    7: 1.8,
    8: 1.7,
    9: 1.5,
    15: 2.1,
    16: 2.0,
    17: 1.8,
    35: 2.0,
    53: 2.2
}

DUDE_PATH = '../data/dude/'

FRAG_PATH = '../data/mol/'


def grid_generator(training_data, batch_size, grid_width, grid_resolution):
    '''
    Generate grids during training
    '''
    while True:
        targets = [None] * batch_size
        ligands = [None] * batch_size
        fragments = [None] * batch_size

        # randomly select a training example
        for i in range(batch_size):
            # pick random target
            (target, ligs) = training_data[np.random.randint(len(training_data))]

            # pick random ligand
            ligand = np.random.choice(ligs)

            # generate grid
            center = ligand.center()

            # g_target = grid.generate_grid_cpu(target.atoms, ATOM_TYPES, VDW_RADIUS, center, grid_width, grid_resolution)
            g_target = grid.generate_grid_fast(target.atoms, ATOM_TYPES, VDW_RADIUS, center, grid_width, grid_resolution)
            # g_ligand = grid.generate_grid_cpu(ligand.atoms + ligand.conn, ATOM_TYPES, VDW_RADIUS, center, grid_width, grid_resolution)
            g_ligand = grid.generate_grid_fast(ligand.atoms + ligand.conn, ATOM_TYPES, VDW_RADIUS, center, grid_width, grid_resolution)
            # g_frag = np.sum(grid.generate_grid_cpu(ligand.frag, ATOM_TYPES, VDW_RADIUS, center, grid_width, grid_resolution), axis=3)
            g_frag = np.sum(grid.generate_grid_fast(ligand.frag, ATOM_TYPES, VDW_RADIUS, center, grid_width, grid_resolution), axis=3)
            g_frag = np.reshape(g_frag, g_frag.shape + (1,))

            # add active to batch
            targets[i] = g_target
            ligands[i] = g_ligand
            fragments[i] = g_frag

        s_targets = np.stack(targets)
        s_ligands = np.stack(ligands)
        s_fragments = np.stack(fragments)

        yield ({
            'ligand_in': s_ligands,
            'target_in': s_targets
        }, {
            'fragment_out': s_fragments
        })

def full_grid_generator(data, grid_width, grid_resolution):
    '''
    Generate a labeled pair for every example in the dataset
    '''

    targets = []
    ligands = []
    fragments = []

    for (target, ligs) in data:
        for ligand in tqdm.tqdm(ligs[:10]):
            center = ligand.center()

            # g_target = grid.generate_grid_cpu(target.atoms, ATOM_TYPES, VDW_RADIUS, center, grid_width, grid_resolution)
            g_target = grid.generate_grid_fast(target.atoms, ATOM_TYPES, VDW_RADIUS, center, grid_width, grid_resolution)
            # g_ligand = grid.generate_grid_cpu(ligand.atoms + ligand.conn, ATOM_TYPES, VDW_RADIUS, center, grid_width, grid_resolution)
            g_ligand = grid.generate_grid_fast(ligand.atoms + ligand.conn, ATOM_TYPES, VDW_RADIUS, center, grid_width, grid_resolution)
            # g_frag = np.sum(grid.generate_grid_cpu(ligand.frag, ATOM_TYPES, VDW_RADIUS, center, grid_width, grid_resolution), axis=3)
            g_frag = np.sum(grid.generate_grid_fast(ligand.frag, ATOM_TYPES, VDW_RADIUS, center, grid_width, grid_resolution), axis=3)
            g_frag = np.reshape(g_frag, g_frag.shape + (1,))

            targets.append(g_target)
            ligands.append(g_ligand)
            fragments.append(g_frag)

    s_targets = np.stack(targets)
    s_ligands = np.stack(ligands)
    s_fragments = np.stack(fragments)

    return (
        {
            'ligand_in': s_ligands,
            'target_in': s_targets
        }, 
        {
            'fragment_out': s_fragments
        }
    )


def load_data(targets):
    '''
    Load data
    '''
    # format:
    # (target, [ligand])
    training_sets = []

    # iterate through targets
    for t in tqdm.tqdm(targets):
        # read pdb
        target_pdb = (pybel.readfile('pdb', os.path.join(DUDE_PATH, t, 'receptor.pdb')).next())

        # load as mol
        target_mol = mol.Mol.from_pybel(target_pdb)

        # load ligands
        ligands = mol.Mol.readfile(os.path.join(FRAG_PATH, (t + '.dat')))

        if len(ligands) == 0:
            continue

        training_sets.append((target_mol, ligands))

    return training_sets
