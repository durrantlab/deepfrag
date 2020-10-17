"""
Contains code for gpu-accelerated grid generation.
"""

import math
import ctypes

import torch
import numba
import numba.cuda
import numpy as np


GPU_DIM = 8


@numba.cuda.jit
def gpu_gridify(grid, atom_num, atom_coords, atom_layers, layer_offset,
                batch_idx, width, res, center, rot):
    """Adds atoms to the grid in a GPU kernel.

    This kernel converts atom coordinate information to 3d voxel information.
    Each GPU thread is responsible for one specific grid point. This function
    receives a list of atomic coordinates and atom layers and simply iterates
    over the list to find nearby atoms and add their effect.

    Voxel information is stored in a 5D tensor of type: BxTxNxNxN where:
        B = batch size
        T = number of atom types (receptor + ligand)
        N = grid width (in gridpoints)

    Each invocation of this function will write information to a specific batch
    index specified by batch_idx. Additionally, the layer_offset parameter can
    be set to specify a fixed offset to add to each atom_layer item.

    How it works:
    1. Each GPU thread controls a single gridpoint. This gridpoint coordinate
        is translated to a "real world" coordinate by applying rotation and
        translation vectors.
    2. Each thread iterates over the list of atoms and checks for atoms within
        a threshold to add to the grid.

    Args:
        grid: DeviceNDArray tensor where grid information is stored
        atom_num: number of atoms
        atom_coords: array containing (x,y,z) atom coordinates
        atom_layers: array containing (idx) offsets that specify which layer to
            store this atom. (-1 can be used to ignore an atom)
        layer_offset: a fixed ofset added to each atom layer index
        batch_idx: index specifiying which batch to write information to
        width: number of grid points in each dimension
        res: distance between neighboring grid points in angstroms
            (1 == gridpoint every angstrom)
            (0.5 == gridpoint every half angstrom, e.g. tighter grid)
        center: (x,y,z) coordinate of grid center
        rot: (x,y,z,y) rotation quaternion
    """
    x,y,z = numba.cuda.grid(3)

    # center around origin
    tx = x - (width/2)
    ty = y - (width/2)
    tz = z - (width/2)

    # scale by resolution
    tx = tx * res
    ty = ty * res
    tz = tz * res

    # apply rotation vector
    aw = rot[0]
    ax = rot[1]
    ay = rot[2]
    az = rot[3]

    bw = 0
    bx = tx
    by = ty
    bz = tz

    # multiply by rotation vector
    cw = (aw * bw) - (ax * bx) - (ay * by) - (az * bz)
    cx = (aw * bx) + (ax * bw) + (ay * bz) - (az * by)
    cy = (aw * by) + (ay * bw) + (az * bx) - (ax * bz)
    cz = (aw * bz) + (az * bw) + (ax * by) - (ay * bx)

    # multiply by conjugate
    # dw = (cw * aw) - (cx * (-ax)) - (cy * (-ay)) - (cz * (-az))
    dx = (cw * (-ax)) + (cx * aw) + (cy * (-az)) - (cz * (-ay))
    dy = (cw * (-ay)) + (cy * aw) + (cz * (-ax)) - (cx * (-az))
    dz = (cw * (-az)) + (cz * aw) + (cx * (-ay)) - (cy * (-ax))

    # apply translation vector
    tx = dx + center[0]
    ty = dy + center[1]
    tz = dz + center[2]

    i = 0
    while i < atom_num:
        # fetch atom
        fx, fy, fz = atom_coords[i]
        ft = atom_layers[i]
        i += 1

        # invisible atoms
        if ft == -1:
            continue

        # fixed radius (^2)
        r2 = 4

        # quick cube bounds check
        if abs(fx-tx) > r2 or abs(fy-ty) > r2 or abs(fz-tz) > r2:
            continue

        # compute squared distance to atom
        d2 = (fx-tx)**2 + (fy-ty)**2 + (fz-tz)**2

        # compute effect
        v = math.exp((-2 * d2) / r2)

        # add effect
        if d2 < r2:
            grid[batch_idx, layer_offset+ft, x, y, z] += v


def mol_gridify(grid, atom_coords, atom_layers, layer_offset, batch_idx,
                width, res, center, rot):
    """Wrapper around gpu_gridify.

    (See gpu_gridify() for details)
    """
    dw = ((width - 1) // GPU_DIM) + 1
    gpu_gridify[(dw,dw,dw), (GPU_DIM,GPU_DIM,GPU_DIM)](
        grid, len(atom_coords), atom_coords, atom_layers, layer_offset,
        batch_idx, width, res, center, rot
    )


def make_tensor(shape):
    """Creates a pytorch tensor and numba array with shared GPU memory backing.

    Args:
        shape: the shape of the array

    Returns:
        (torch_arr, cuda_arr)
    """
    # get cuda context
    ctx = numba.cuda.cudadrv.driver.driver.get_active_context()

    # setup tensor on gpu
    t = torch.zeros(size=shape, dtype=torch.float32).cuda()

    memory = numba.cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel() * 4)
    cuda_arr = numba.cuda.cudadrv.devicearray.DeviceNDArray(
        t.size(),
        [i*4 for i in t.stride()],
        np.dtype('float32'),
        gpu_data=memory,
        stream=torch.cuda.current_stream().cuda_stream
    )

    return (t, cuda_arr)


def rand_rot():
    """Returns a random uniform quaternion rotation."""
    q = np.random.normal(size=4) # sample quaternion from normal distribution
    q = q / np.sqrt(np.sum(q**2)) # normalize
    return q


def get_batch(data, rec_channels, parent_channels, batch_size=16, batch_set=None,
              width=48, res=0.5, ignore_receptor=False, ignore_parent=False):
    """Builds a batch grid from a FragmentDataset.

    Args:
        data: a FragmentDataset object
        rec_channels: number of receptor channels
        parent_channels: number of parent channels
        batch_size: size of the batch
        batch_set: if not None, specify a list of data indexes to use for each
            item in the batch
        width: grid width
        res: grid resolution
        ignore_receptor: if True, ignore receptor atoms
        ignore_parent: if True, ignore parent atoms

    Returns: (torch_grid, batch_set)
        torch_grid: pytorch Tensor with voxel information
        examples: list of examples used
    """
    assert (not (ignore_receptor and ignore_parent)), "Can't ignore parent and receptor!"

    dim = 0
    if not ignore_receptor:
        dim += rec_channels
    if not ignore_parent:
        dim += parent_channels

    # create a tensor with shared memory on the gpu
    torch_grid, cuda_grid = make_tensor((batch_size, dim, width, width, width))

    if batch_set is None:
        batch_set = np.random.choice(len(data), size=batch_size, replace=False)

    examples = [data[idx] for idx in batch_set]

    for i in range(len(examples)):
        example = examples[i]
        rot = rand_rot()

        if ignore_receptor:
            mol_gridify(
                cuda_grid,
                example['p_coords'],
                example['p_types'],
                layer_offset=0,
                batch_idx=i,
                width=width,
                res=res,
                center=example['conn'],
                rot=rot
            )
        elif ignore_parent:
            mol_gridify(
                cuda_grid,
                example['r_coords'],
                example['r_types'],
                layer_offset=0,
                batch_idx=i,
                width=width,
                res=res,
                center=example['conn'],
                rot=rot
            )
        else:
            mol_gridify(
                cuda_grid,
                example['p_coords'],
                example['p_types'],
                layer_offset=0,
                batch_idx=i,
                width=width,
                res=res,
                center=example['conn'],
                rot=rot
            )
            mol_gridify(
                cuda_grid,
                example['r_coords'],
                example['r_types'],
                layer_offset=parent_channels,
                batch_idx=i,
                width=width,
                res=res,
                center=example['conn'],
                rot=rot
            )

    return torch_grid, examples


def get_raw_batch(r_coords, r_types, p_coords, p_types, conn, num_samples=32,
                  width=24, res=1, rec_channels=9, parent_channels=9):
    """Sample a raw batch with provided atom coordinates.

    Args:
        r_coords: receptor coordinates
        r_types: receptor types (layers)
        p_coords: parent coordinates
        p_types: parent types (layers)
        conn: (x,y,z) connection point
        num_samples: number of rotations to sample
        width: grid width
        res: grid resolution
        rec_channels: number of receptor channels
        parent_channels: number of parent chanels
    """
    B = num_samples
    T = rec_channels + parent_channels
    N = width

    torch_grid, cuda_grid = make_tensor((B,T,N,N,N))

    for i in range(num_samples):
        rot = rand_rot()
        mol_gridify(cuda_grid, p_coords, p_types, layer_offset=0, batch_idx=i,
                        width=width, res=res, center=conn, rot=rot)
        mol_gridify(cuda_grid, r_coords, r_types, layer_offset=p_dim, batch_idx=i,
                        width=width, res=res, center=conn, rot=rot)

    return torch_grid
