import numpy as np
from ahrs.filters import Madgwick
from ahrs.common import DCM

def unit_vector(v):
    '''Normalizes a vector over its norm. It works for both (N x 1) and (N x m) vectors.'''
    v_n = np.zeros_like(v)
    if (len(v) > 1) & (v.ndim > 1):
        for i in range(len(v)):
            v_n[i] = v[i] / np.linalg.norm(v[i])
    
    elif (len(v) > 1) & (v.ndim == 1):
        v_n = v / np.linalg.norm(v)
    
    return v_n

def rodrigues_rotation(g_sensor, g_global):
    v = g_sensor / np.linalg.norm(g_sensor)
    t = g_global / np.linalg.norm(g_global)

    axis = np.cross(v, t)
    axis_norm = np.linalg.norm(axis)

    angle = np.arctan2(axis_norm, np.dot(v, t))
    axis = axis / axis_norm

    K = np.array([[     0, -axis[2],  axis[1]],
              [ axis[2],      0, -axis[0]],
              [-axis[1],  axis[0],     0]])

    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K @ K)

    return R


def get_linear_acceleration(a_m=None, w_m=None, fs=None, g_global=None):
    # Compute overall orientation over time
    mad = Madgwick(gyr=w_m, acc=a_m, frequency=fs)
    q = mad.Q # Orientation quaternion

    # Initialize acceleration of the inertial frame
    a_I = np.zeros_like(a_m)
    
    for k in range(q.shape[0]):
        R_IB = DCM()
        R_IB = R_IB.from_quaternion(mad.Q[k])
        a_I[k] = R_IB @ a_m[k] - g_global

    return a_I
