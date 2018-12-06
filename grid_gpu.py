
import math

from numba import cuda
import numpy as np

import rotation


# support (8,8,8) invocations
GPU_DIM = 8


@cuda.jit
def add_atoms(grid, atom_coords, atom_layer, atom_vdw, num_atoms, grid_span):
    """
    GPU Kernel invoked on each (x,y,z) point in the grid
    """
    x,y,z = cuda.grid(3)
    
    # out of bounds check
    if x < 0 or y < 0 or z < 0 or x >= grid_span or y >= grid_span or z >= grid_span:
            return

    # iterate over atoms
    for i in range(num_atoms):
        coords = atom_coords[i]
        layer = atom_layer[i]
        vdw = atom_vdw[i]
        
        dist = math.sqrt((x - coords[0])**2 + (y - coords[1])**2 + (z - coords[2])**2)
        
        d = dist
        r = vdw

        # smoothed gaussian equation
        if d < r:
            v = math.exp((-2 * (d**2)) / (r**2))
        elif r <= d <= (1.5 * r):
            v = ((4 / (math.exp(2.0) * (r**2))) * (d**2)) - ((12 / (math.exp(2.0) * r)) * d) + (9 / math.exp(2.0))
        else:
            v = 0
    
        # add grid effect
        grid[x,y,z,int(layer)] += v


def generate_grid(
    atoms,                   
    atom_types, 
    vdw_radius, 
    center, 
    width, 
    resolution, 
    rotation_vec=np.array([1,0,0,0]), 
    translation_vec=np.array([0,0,0])
    ):
    """
    Generate a grid for a molecule on the gpu.

    atoms: list of (x,y,z,atom_type)
    vdw_radius: mapping from atom type to van der waals radius
    center: (x,y,z) position of the grid center
    width: length of one dimension of the grid (in angstroms)
    resolution: grid resolution (in angstroms)
    
    Optional parameters:
    rotation_vec: a quaternion (w,x,y,z) describing a rotation vector about the grid center
    translation_vec: a tuple (x,y,z) describing a translation (in angstroms)

    Returns a numpy matrix of shape (S,S,S,T) where S = (width / resolution) and T = len(atom_types)
    """
    # indirect map atom_type -> layer
    atom_indirect = {atom_types[x]:x for x in range(len(atom_types))}

    # effective size of the grid
    grid_span = int(width / resolution)

    # define an empty grid
    grid = np.zeros((grid_span, grid_span, grid_span, len(atom_types)))

    # filter atoms in atom_types
    atoms = [atom for atom in atoms if atom[3] in atom_types]
    
    # prepare data
    atom_coords = np.zeros((len(atoms), 3))
    atom_layer = np.zeros(len(atoms))
    atom_vdw = np.zeros(len(atoms))

    # iterate through atoms
    for i in range(len(atoms)):
        atom = atoms[i]
        
        # get atom type
        atom_type = atom[3]

        # get layer
        layer = atom_indirect[atom_type]

        # get location
        coords = np.array(atom[:3])

        #######################
        # Geometry Conversion #
        #######################

        # normalize to center
        coords_normal = coords - center
        
        # apply rotation
        coords_rotated = rotation.apply_rot(coords_normal, rotation_vec)
        
        # apply translation
        coords_translated = coords_rotated + translation_vec

        # scale by resolution
        coords_scaled = coords_translated / resolution

        # shift to grid space
        coords_grid = coords_scaled + (grid_span/2)

        # van-der-waals radius
        vdw = vdw_radius[atom_type]

        # scale by resolution
        vdw_scaled = vdw / resolution

        atom_coords[i] = coords_grid
        atom_layer[i] = layer
        atom_vdw[i] = vdw_scaled

    dx = int(np.ceil(grid_span / GPU_DIM))
    dy = int(np.ceil(grid_span / GPU_DIM))
    dz = int(np.ceil(grid_span / GPU_DIM))

    # invoke the GPU kernel
    add_atoms[(dx,dy,dz), (GPU_DIM,GPU_DIM,GPU_DIM)](grid, atom_coords, atom_layer, atom_vdw, len(atoms), grid_span)

    return grid
