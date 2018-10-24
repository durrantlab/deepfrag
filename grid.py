

import numpy as np


def smoothed_gaussian(d,r):
    '''
    Continuous piecwise gaussian from 0 to r that goes to zero from
    r to r*1.5.

    Equation 1 in Protein-Ligand Scoring with Convolutional Neural Networks
    '''
    if d < r:
        return np.exp((-2 * (d**2)) / (r**2))
    elif r <= d <= (1.5 * r):
        return ((4 / (np.exp(2) * (r**2))) * (d**2)) - ((12 / (np.exp(2) * r)) * d) + (9 / np.exp(2))
    else:
        return 0

def generate_grid_cpu(atoms, atom_types, vdw_radius, center, width, resolution):
    '''
    Generate a grid for a given molecule.

    atoms: list of (x,y,z,atom_type)
    vdw_radius: mapping from atom type to van der waals radius
    center: (x,y,z) position of the grid center
    width: length of one dimension of the grid (in angstroms)
    resolution: grid resolution (in angstroms)

    Returns a numpy matrix of shape (S,S,S,T) where S = (width / resolution) and T = len(atom_types)
    '''
    # indirect map atom_type -> layer
    atom_indirect = {atom_types[x]:x for x in range(len(atom_types))}

    # effective size of the grid
    grid_span = int(width / resolution)

    # define an empty grid
    grid = np.zeros((grid_span, grid_span, grid_span, len(atom_types)))

    # bounding corners
    g_min = np.array(center) - (width / 2.0)
    g_max = np.array(center) + (width / 2.0)

    # iterate through atoms
    for atom in atoms:
        # get atom type
        atom_type = atom[3]
        if atom_type not in atom_types:
            continue

        # get layer
        layer = atom_indirect[atom_type]

        # get location
        coords = np.array(atom[:3])

        #######################
        # Geometry Conversion #
        #######################

        # normalize to center
        coords_normal = coords - center

        # scale by resolution
        coords_scaled = coords_normal / resolution

        # shift to grid space
        coords_grid = coords_scaled + (grid_span/2)

        # van-der-waals radius
        vdw = vdw_radius[atom_type]

        # scale by resolution
        vdw_scaled = vdw / resolution

        # bounding box of vdw radius in grid space
        vdw_min = np.floor(coords_grid - vdw_scaled).astype(int)
        vdw_max = np.ceil(coords_grid + vdw_scaled + 1).astype(np.int)

        # multidimensional iteration
        for off in np.ndindex(tuple(vdw_max - vdw_min)):
        # for off in [np.round(coords_grid - vdw_min).astype(int)]:
            # print(off)
            idx = vdw_min + off
            [x,y,z] = idx
            # print(idx)

            if x < 0 or y < 0 or z < 0 or x >= grid_span or y >= grid_span or z >= grid_span:
                continue

            # compute distance
            dist = np.linalg.norm((coords_grid - idx) * resolution)

            # compute value
            v = smoothed_gaussian(dist, vdw)

            try:
                # add grid effect
                grid[x,y,z,layer] += v
            except Exception as e:
                print(e)
                pass

    return grid
