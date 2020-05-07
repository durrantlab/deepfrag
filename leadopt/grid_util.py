'''
grid_util.py

contains code for gpu-accelerated grid generation
'''

import math
import ctypes

import torch
import numba
import numba.cuda
import numpy as np


GPU_DIM = 8


@numba.cuda.jit
def gpu_gridify(grid, width, res, center, rot, atom_num, atom_coords, atom_types, layer_offset, batch_i):
    '''
    GPU kernel to add atoms to a grid
    
    width, res, offset and rot control the grid view
    '''
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
        ft = atom_types[i][0]
        i += 1
        
        # invisible atoms
        if ft == -1:
            continue
        
        r2 = 4
        
        # exit early
        if abs(fx-tx) > r2 or abs(fy-ty) > r2 or abs(fz-tz) > r2:
            continue
        
        # compute squared distance to atom
        d2 = (fx-tx)**2 + (fy-ty)**2 + (fz-tz)**2
        
        # compute effect
        v = math.exp((-2 * d2) / r2)
        
        # add effect
        if d2 < r2:
            grid[batch_i,layer_offset+ft,x,y,z] += v


def make_tensor(shape):
    '''
    Create a pytorch tensor and numba array with the same GPU memory backing
    '''
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
    '''
    Sample a random 3d rotation from a uniform distribution
    
    Returns a quaternion vector (w,x,y,z)
    '''
    q = np.random.normal(size=4) # sample quaternion from normal distribution
    q = q / np.sqrt(np.sum(q**2)) # normalize
    return q


def mol_gridify(
    grid,
    mol_atoms, 
    mol_types, 
    batch_i,
    center=np.array([0,0,0]),
    width=48, 
    res=0.5,
    rot=np.array([1,0,0,0]),
    layer_offset=0,
    ):
    '''wrapper to invoke gpu gridify kernel'''
    dw = ((width - 1) // GPU_DIM) + 1
    
    gpu_gridify[(dw,dw,dw), (GPU_DIM,GPU_DIM,GPU_DIM)](
        grid, 
        width, 
        res, 
        center,
        rot,
        len(mol_atoms), 
        mol_atoms,
        mol_types,
        layer_offset,
        batch_i
    )
    

def get_batch(data, batch_set=None, batch_size=16, width=48, res=0.5, ignore_receptor=False, ignore_parent=False, include_freq=False):
    
    assert (not (ignore_receptor and ignore_parent)), "Can't ignore parent and receptor!"

    dim = 18
    if ignore_receptor or ignore_parent:
        dim = 9

    # create a tensor with shared memory on the gpu
    t, grid = make_tensor((batch_size, dim, width, width, width))
    
    if batch_set is None:
        batch_set = np.random.choice(len(data), size=batch_size, replace=False)
    
    fingerprints = np.zeros((batch_size, data.fingerprints['fingerprint_data'].shape[1]))
    freq = np.zeros(batch_size)
    
    for i in range(len(batch_set)):
        idx = batch_set[i]
        f_coords, f_types, p_coords, p_types, r_coords, r_types, conn, fp, extra = data[idx]
        
        # random rotation
        rot = rand_rot()
        
        if ignore_receptor:
            mol_gridify(grid, p_coords, p_types, batch_i=i, center=conn, width=width, res=res, rot=rot, layer_offset=0)
        elif ignore_parent:
            mol_gridify(grid, r_coords, r_types, batch_i=i, center=conn, width=width, res=res, rot=rot, layer_offset=0)
        else:
            mol_gridify(grid, p_coords, p_types, batch_i=i, center=conn, width=width, res=res, rot=rot, layer_offset=0)
            mol_gridify(grid, r_coords, r_types, batch_i=i, center=conn, width=width, res=res, rot=rot, layer_offset=9)
        
        fingerprints[i] = fp
        freq[i] = extra['freq']
        
    t_fingerprints = torch.Tensor(fingerprints).cuda()
    t_freq = torch.Tensor(freq).cuda()
    
    if include_freq:
        return t, t_fingerprints, t_freq, batch_set
    else:
        return t, t_fingerprints, batch_set


def get_batch_dual(data, batch_set=None, batch_size=16, width=48, res=0.5, ignore_receptor=False, ignore_parent=False):

    # get batch
    t, fp, batch_set = get_batch(data, batch_set, batch_size, width, res, ignore_receptor, ignore_parent)

    f = data.fingerprints['fingerprint_data']

    # corrupt fingerprints
    false_fp = torch.clone(fp)
    for i in range(batch_size):
        # idx = np.random.randint(fp.shape[1])
        # false_fp[i,idx] = (1 - false_fp[i,idx]) # flip
        idx = np.random.randint(f.shape[0])
        false_fp[i] = torch.Tensor(f[idx]) # replace

    comb_t = torch.cat([t,t], axis=0)
    comb_fp = torch.cat([fp, false_fp], axis=0)

    y = torch.zeros((batch_size * 2,1)).cuda()
    y[:batch_size] = 1

    return (comb_t, comb_fp, y, batch_set)


def get_batch_full(data, batch_set=None, batch_size=16, width=48, res=0.5, ignore_receptor=False, ignore_parent=False):
    
    assert (not (ignore_receptor and ignore_parent)), "Can't ignore parent and receptor!"

    dim = 18
    if ignore_receptor or ignore_parent:
        dim = 9

    # create a tensor with shared memory on the gpu
    t_context, grid_context = make_tensor((batch_size, dim, width, width, width))
    t_frag, grid_frag = make_tensor((batch_size, 9, width, width, width))
    
    if batch_set is None:
        batch_set = np.random.choice(len(data), size=batch_size, replace=False)
    
    for i in range(len(batch_set)):
        idx = batch_set[i]
        f_coords, f_types, p_coords, p_types, r_coords, r_types, conn, fp = data[idx]
        
        # random rotation
        rot = rand_rot()
        
        if ignore_receptor:
            mol_gridify(grid_context, p_coords, p_types, batch_i=i, center=conn, width=width, res=res, rot=rot, layer_offset=0)
        elif ignore_parent:
            mol_gridify(grid_context, r_coords, r_types, batch_i=i, center=conn, width=width, res=res, rot=rot, layer_offset=0)
        else:
            mol_gridify(grid_context, p_coords, p_types, batch_i=i, center=conn, width=width, res=res, rot=rot, layer_offset=0)
            mol_gridify(grid_context, r_coords, r_types, batch_i=i, center=conn, width=width, res=res, rot=rot, layer_offset=9)
        
        mol_gridify(grid_frag, f_coords, f_types, batch_i=i, center=conn, width=width, res=res, rot=rot, layer_offset=0)

    return t_context, t_frag, batch_set


def get_raw_batch(r_coords, r_types, p_coords, p_types, conn, num_samples=32, width=24, res=1, r_dim=9, p_dim=9):
    
    # create a tensor with shared memory on the gpu
    t, grid = make_tensor((num_samples, (r_dim + p_dim), width, width, width))
    
    for i in range(num_samples):
        # random rotation
        rot = rand_rot()
        
        mol_gridify(grid, p_coords, p_types, batch_i=i, center=conn, width=width, res=res, rot=rot, layer_offset=0)
        mol_gridify(grid, r_coords, r_types, batch_i=i, center=conn, width=width, res=res, rot=rot, layer_offset=p_dim)
        
    return t
