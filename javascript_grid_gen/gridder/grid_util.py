"""
Contains code for gpu-accelerated grid generation.
"""

import math

GPU_DIM = 8


def grid_to_real_no_rot(x, y, z, half_width, res, center):  # , rot):
    """Convert a grid (x,y,z) coordinate to real world coordinate."""

    return (
        grid_to_real_no_rot_one_axis(x, half_width, res, center.x),
        grid_to_real_no_rot_one_axis(y, half_width, res, center.y),
        grid_to_real_no_rot_one_axis(z, half_width, res, center.z),
    )


def grid_to_real_no_rot_one_axis(val_axis, half_width, res, center_axis):  # , rot):
    """Convert a grid value along an axis coordinate to real world coordinate
    along that axis."""

    # Center around origin.
    tval_axis = val_axis - half_width

    # Scale by resolution.
    tval_axis = tval_axis * res

    # Apply translation vector.
    tval_axis = tval_axis + center_axis  # [0]

    return tval_axis


def filter_atoms_no_rot(atom_coords, atom_layers, width, res, center):
    """Filter atoms based on a grid bounding box.

    Returns (filt_coords, filt_layers).
    """
    PAD = 1.75

    # Compute grid extremes.
    half_width = width / 2
    ax, ay, az = grid_to_real_no_rot(0, 0, 0, half_width, res, center)
    bx, by, bz = grid_to_real_no_rot(
        width - 1, width - 1, width - 1, half_width, res, center
    )

    filt_coords = []
    filt_layers = []

    for i in range(len(atom_coords)):
        x, y, z = atom_coords[i]

        # TODO: this bounds check only works for a rotation vector of: [1,0,0,0]
        # An alternative approach is to define real_to_grid and compute the
        # nearest gridpoint for each atom.
        if (
            x > ax - PAD
            and y > ay - PAD
            and z > az - PAD
            and x < bx + PAD
            and y < by + PAD
            and z < bz + PAD
        ):
            filt_coords.append(atom_coords[i])
            filt_layers.append(atom_layers[i])

    return filt_coords, filt_layers


def mol_gridify2_one_channel(
    grid,
    layer_offset,
    abs_channel_to_consider,
    width,
    res,
    center,
    filt_coords,
    filt_layers,
):

    # Fixed atom radius (r = 1.75).
    r = 1.75
    r2 = r * r
    half_width = width / 2

    # Now compare the grid point location to each atom position.
    for i in range(len(filt_coords)):
        ft_with_offset = layer_offset + filt_layers[i]
        if ft_with_offset != abs_channel_to_consider:
            continue

        nx, ny, nz = filt_coords[i]

        # Iterate over grid points.
        rng = range(width)
        for x in rng:
            # Compute the effective grid point position.
            tx = grid_to_real_no_rot_one_axis(x, half_width, res, center.x)
            d2_x_prt = (nx - tx) ** 2
            for y in rng:
                # Compute the effective grid point position.
                ty = grid_to_real_no_rot_one_axis(y, half_width, res, center.y)
                d2_x_prt_plus_d2_y_prt = d2_x_prt + (ny - ty) ** 2
                for z in rng:
                    # Compute the effective grid point position.
                    tz = grid_to_real_no_rot_one_axis(z, half_width, res, center.z)

                    # Distance squared.
                    d2 = d2_x_prt_plus_d2_y_prt + (nz - tz) ** 2

                    if d2 > r2:
                        continue

                    # TODO: if we implement multi-sample grids, replace 0 with
                    # batch_index or similar.
                    grid[0][ft_with_offset][x][y][z] += math.exp((-2 * d2) / r2)

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


def get_raw_batch_one_channel(
    r_coords,
    r_types,
    p_coords,
    p_types,
    conn,
    num_samples,  # =3, VAL: 1
    width,  # =24, VAL: 24
    res,  # =0.5, VAL: 0.75
    channel,
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
        channel: the channel to consider
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

    B = num_samples  # 1
    T = rec_channels + parent_channels  # 9
    N = width  # 24

    shape = (B, T, N, N, N)
    grid = make_tensor(shape)

    # for i in range(num_samples):
    # rot = rand_rot()

    if channel < parent_channels:
        coords = p_coords
        types = p_types
        offset = 0
    else:
        coords = r_coords
        types = r_types
        offset = parent_channels

    # Filter atoms based on the grid bounding box.
    filt_coords, filt_layers = filter_atoms_no_rot(coords, types, width, res, conn)

    # for i in range(parent_channels):
    # print(i)
    grid = mol_gridify2_one_channel(
        grid, offset, channel, width, res, conn, filt_coords, filt_layers
    )

    return flatten_tensor(grid, shape)
