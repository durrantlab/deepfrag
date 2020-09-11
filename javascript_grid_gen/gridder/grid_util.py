"""
Contains code for gpu-accelerated grid generation.
"""

import math
import random

# __pragma__ ('skip')
from .kdtrees import KDTree

# __pragma__ ('noskip')

"""?
from .gridder.kdtrees._kdtree import KDTree
?"""

GPU_DIM = 8


def coord_to_key(c: list) -> int:
    """Converts a 3D coordinate to a hash.

    :param c: The 3D coordinates.
    :type c: list
    :return: The hash.
    :rtype: int
    """
    return 1000 * round(c[0] + 1000 * (c[1] + c[2] * 1000))


def mol_gridify(
    grid,
    atom_coords,
    atom_layers,
    layer_offset,
    num_layers_to_consider,
    width,
    res,
    center,
    rot,
):
    """Adds atoms to the tensor. Converts atom coordinate information to 3d
    voxel information. This function receives a list of atomic coordinates and
    atom layers and simply iterates over the list to find nearby atoms and add
    their effect.

    Voxel information is stored in a 5D tensor of type: BxTxNxNxN where:
        B = batch size
        T = number of atom types (receptor + ligand)
        N = grid width (in gridpoints)

    The layer_offset parameter can be set to specify a fixed offset to add to
    each atom_layer item.

    Args:
        grid: Tensor where grid information is stored
        atom_coords: array containing (x,y,z) atom coordinates
        atom_layers: array containing (idx) offsets that specify which layer to
            store this atom. (-1 can be used to ignore an atom)
        layer_offset: a fixed ofset added to each atom layer index
        num_layers_to_consider: index specifiying which batch to write
            information to
        width: number of grid points in each dimension
        res: distance between neighboring grid points in angstroms
            (1 == gridpoint every angstrom)
            (0.5 == gridpoint every half angstrom, e.g. tighter grid)
        center: (x,y,z) coordinate of grid center
        rot: (x,y,z,y) rotation quaternion
    """

    # r2 = 4  # fixed radius (^2)
    r2 = 3.0625  # TODO: Or is it 1.75 ^ 2 = 3.0625
    half_width = width / 2

    # TODO: Remove cruft below. Leaving here for now in case you need it.

    # Organize the data by layer.
    # data = {}
    # for i, atom_layer in enumerate(atom_layers):
    #     if not atom_layer in data:
    #         data[atom_layer] = []
    #     atom_coord = atom_coords[i]
    #     data[atom_layer].append([atom_coord.x, atom_coord.y, atom_coord.z])

    # Organize the data by layer.
    # data = {}
    # for i, atom_layer in enumerate(atom_layers):
    #     if not atom_layer in data:
    #         data[atom_layer] = []
    #     atom_coord = atom_coords[i]
    #     data[atom_layer].append([atom_coord.x, atom_coord.y, atom_coord.z])

    # Replace with a tree
    # for i in data.keys():
    #     coors = data[i]
    #     data[i] = KDTree.initialize(coors, 3, 0, None)

    # First generate a kdtree of the atom coordinates. To speed calculate on
    # CPU. Using pure-python KDTrees implementation for conversion to
    # javascript later.
    data = [[c.x, c.y, c.z] for i, c in enumerate(atom_coords)]
    tree = KDTree.initialize(data, 3, 0, None)

    # Make a dictionary to map the coordinates (hash) to the atom layer for
    # fast lookup later.
    layers = {}
    for i, c in enumerate(data):
        layers[coord_to_key(c)] = atom_layers[i]

    for rotation_idx in range(len(grid)):  # Rotation
        for layer_idx in range(len(grid[rotation_idx])):
            if layer_idx < layer_offset:
                continue
            if layer_idx >= layer_offset + num_layers_to_consider:
                continue

            print(rotation_idx, layer_idx)  # To indicate progress.


            for x in range(len(grid[rotation_idx][layer_idx])):
                for y in range(len(grid[rotation_idx][layer_idx][x])):
                    for z in range(len(grid[rotation_idx][layer_idx][y])):
                        # center around origin
                        tx = x - half_width
                        ty = y - half_width
                        tz = z - half_width

                        # scale by resolution
                        tx = tx * res
                        ty = ty * res
                        tz = tz * res

                        # apply rotation vector
                        aw = rot[0]
                        ax = rot[1]
                        ay = rot[2]
                        az = rot[3]

                        # bw = 0
                        bx = tx
                        by = ty
                        bz = tz

                        # multiply by rotation vector
                        # cw = (aw * bw) - (ax * bx) - (ay * by) - (az * bz)
                        # cx = (aw * bx) + (ax * bw) + (ay * bz) - (az * by)
                        # cy = (aw * by) + (ay * bw) + (az * bx) - (ax * bz)
                        # cz = (aw * bz) + (az * bw) + (ax * by) - (ay * bx)

                        cw = -(ax * bx) - (ay * by) - (az * bz)
                        cx = (aw * bx) + (ay * bz) - (az * by)
                        cy = (aw * by) + (az * bx) - (ax * bz)
                        cz = (aw * bz) + (ax * by) - (ay * bx)

                        # multiply by conjugate
                        # dw = (cw * aw) - (cx * (-ax)) - (cy * (-ay)) - (cz * (-az))
                        dx = (cw * (-ax)) + (cx * aw) + (cy * (-az)) - (cz * (-ay))
                        dy = (cw * (-ay)) + (cy * aw) + (cz * (-ax)) - (cx * (-az))
                        dz = (cw * (-az)) + (cz * aw) + (cx * (-ay)) - (cy * (-ax))

                        # apply translation vector
                        tx = dx + center.x  # [0]
                        ty = dy + center.y  # [1]
                        tz = dz + center.z  # [2]

                        # Get the closest atom.
                        pt = [tx, ty, tz]
                        closest_atom_coords = tree.nearest_neighbor(
                            pt, 1, []
                        )

                        # If the closest one is farther than r2 away,
                        # continue.
                        if closest_atom_coords[0][1] > r2:
                            continue

                        # Get other closest atoms (expanding out).
                        closest_atom_coords = tree.proximal_neighbor(pt, r2, [])

                        for atom_inf in closest_atom_coords:
                            # Get the atom type.
                            ft = layers[coord_to_key(atom_inf[0])]

                            # invisible atoms
                            if ft == -1:
                                continue

                            # fx, fy, fz = atom_inf[0]  # atom_coords[i]
                            # i += 1

                            # quick cube bounds check. TODO: not needed now
                            # that using kdtree?
                            # if (
                            #     abs(fx - tx) > r2
                            #     or abs(fy - ty) > r2
                            #     or abs(fz - tz) > r2
                            # ):
                            #     continue

                            # compute squared distance to atom
                            d2 = atom_inf[1]

                            # compute effect
                            # Point type: 0 (effect(d,r) = exp((-2 * d^2) / r^2))
                            v = math.exp((-2 * d2) / r2)

                            # add effect
                            # Acc type: 0 (sum overlapping points)
                            grid[rotation_idx][layer_offset + ft][x][y][z] += v
    return grid

def flatten_tensor(grid, shape):
    flat = []
    for i1 in range(shape[0]):
        for i2 in range(shape[1]):
            for i3 in range(shape[2]):
                for i4 in range(shape[3]):
                    for i5 in range(shape[4]):
                        flat.append(grid[i1][i2][i3][i4][i5])
    return flat

def make_tensor(shape):
    """Creates a tensor to store the grid data in.

    Args:
        shape: the shape of the tensor

    Returns:
        The tensor.
    """

    t = []
    for i1 in range(shape[0]):
        t1 = []
        for i2 in range(shape[1]):
            t2 = []
            for i3 in range(shape[2]):
                t3 = []
                for i4 in range(shape[3]):
                    t4 = []
                    for i5 in range(shape[4]):
                        t4.append(0)
                    t3.append(t4)
                t2.append(t3)
            t1.append(t2)
        t.append(t1)

    return t


def rand_rot():
    """Returns a random uniform quaternion rotation."""
    # Below if random.
    # q = [random.random() for i in range(4)]
    # l = math.sqrt(sum([v ** 2 for v in q]))
    # q = [v / l for v in q]

    # Below if not random
    q = [1, 0, 0, 0]

    return q


def get_raw_batch(
    r_coords,
    r_types,
    p_coords,
    p_types,
    conn,
    num_samples,  # =3,
    width,  # =24,
    res,  # =0.5,
):
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
    """

    num_samples = 1 if num_samples is None else num_samples  # default was 3
    width = 24 if width is None else width
    res = 0.5 if res is None else res

    # Channels (index -> what is in the channel):
    # 0: parent carbon
    # 1: parent nitrogen
    # 2: parent oxygen
    # 3: parent other (not including hydrogen)
    # 4: receptor carbon
    # 5: receptor nitrogen
    # 6: receptor oxygen
    # 7: receptor sulfur
    # 8: receptor other (not including hydrogen)

    parent_channels = 4  # len(set(p_types))
    rec_channels = 5  # len(set(r_types))

    # TODO: For debugging
    # num_samples = 1

    B = num_samples
    T = rec_channels + parent_channels
    N = width

    shape = (B, T, N, N, N)
    grid = make_tensor(shape)

    # for i in range(num_samples):
    rot = rand_rot()

    grid = mol_gridify(
        grid, p_coords, p_types, 0, parent_channels, width, res, conn, rot,
    )

    # TODO: Harrison should check. parent_channels was p_dim.
    grid = mol_gridify(
        grid, r_coords, r_types, parent_channels, rec_channels, width, res, conn, rot,
    )

    return flatten_tensor(grid, shape)
