# This file contains some code to deal with 3d rotations

import numpy as np


def rand_rot():
    """
    Sample a random 3d rotation from a uniform distribution
    
    Returns a quaternion vector (w,x,y,z)
    """
    q = np.random.normal(size=4) # sample quaternion from normal distribution
    q = q / np.sqrt(np.sum(x ** 2 for x in q)) # normalize
    return tuple(q)


def mult_q_q(a, b):
    """
    Multiply two quaternions
    
    Returns a quaternion vector (w,x,y,z)
    """
    w = (a[0] * b[0]) - (a[1] * b[1]) - (a[2] * b[2]) - (a[3] * b[3])
    x = (a[0] * b[1]) + (a[1] * b[0]) + (a[2] * b[3]) - (a[3] * b[2])
    y = (a[0] * b[2]) + (a[2] * b[0]) + (a[3] * b[1]) - (a[1] * b[3])
    z = (a[0] * b[3]) + (a[3] * b[0]) + (a[1] * b[2]) - (a[2] * b[1])
    return (w,x,y,z)


def apply_rot(vec, q):
    """
    Apply a rotation quaternion to a given vector
    
    Returns a new vector (x,y,z)
    """
    vec_q = (0,) + tuple(vec) # convert vector to quaternion
    conj_q = (q[0], -q[1], -q[2], -q[3]) # calculate conjugate
    vec_q = mult_q_q(q, vec_q)
    vec_q = mult_q_q(vec_q, conj_q)
    
    return vec_q[1:]
