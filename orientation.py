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

def detect_ideal_gravity(g_sensor=None):
    '''Detects the ideal direction of the global gravity axis, accounting for sign.
    The input must be a (3,) np.array containing a measure of the still accelerometer you want to align.'''
    
    g_global = np.zeros(3) # Initialize g_global
    g_unit = unit_vector(g_sensor) 

    # Find where the most prominent axis is, irrespective of the sign for now
    main_axis_idx = np.argmax(np.abs(g_unit)) 

    # Replace that index in g_global with 1, accounting for sign
    g_global[main_axis_idx] = 1 * np.sign(g_unit[main_axis_idx])
    return g_global

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


def get_linear_acceleration(a_m=None, w_m=None, fs=None, g_global=None, convert_to_mss=True):
    '''Removes the gravitational component of the acceleration.'''

    # Measurement unit check (Madgwick wants m/s2 as input)
    if convert_to_mss:
        a_m = a_m * 9.81
        g_global = g_global * 9.81
    
    # Compute overall orientation over time
    mad = Madgwick(gyr=w_m, acc=a_m, frequency=fs, q0=np.array([1, 0, 0, 0]))
    q = mad.Q # Orientation quaternion

    # Initialize acceleration of the inertial frame
    a_I = np.zeros_like(a_m)
    
    for k in range(q.shape[0]):
        R_IB = DCM()
        R_IB = R_IB.from_quaternion(q=q[k])
        a_I[k] = (R_IB @ a_m[k]) # - g_global

    # Convert back to [g] units
    if convert_to_mss:
        a_I /= 9.81

    return a_I
