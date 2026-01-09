import numpy as np
from Orientation.Vectors import unit_vector
from ahrs.filters import Madgwick
from ahrs.common import DCM

def rodrigues_rotation(g_sensor=None, g_global=None):
    '''
    Numerical implementation of the Rodrigues' rotation formula [1]. It creates a rotation matrix
    that maps the orientation of two vectors. It is used here to map the vertical axis of the
    sensor (usually when the person wearing it is standing still) to the ideal gravity axis.

    Inputs:
        - g_sensor (np.array (3,)): the direction of the average acceleromter when the person is standing still
        - g_global (np.arrat (3,)): the direction of ideal gravity

    Returns:
        - R (np.ndarray (3,3)): the rotation matrix given by the Rodrigues' rotation formula

    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    '''
    # Normalize vectors
    v1 = unit_vector(g_sensor)
    v2 = unit_vector(g_global)

    # Cos and Sin and angle
    cos = np.dot(v1, v2) # dot-product of 2 unit vectors is cos of their angle
    sin = np.linalg.norm(np.cross(v1, v2)) # norm of cross-product of 2 unit vectors is sin of their angle
    theta = np.arctan2(sin, cos) # atan2 account for sin and cos ambiguities (correct quadrant)

    # Skew-symmetric vector
    v = unit_vector(np.cross(v1, v2))
    # Skew-symmetric matrix
    K = np.array([[     0, -v[2],  v[1]],
                 [ v[2],      0, -v[0]],
                 [-v[1],  v[0],     0]])

    # Rodrigues' Rotation Formula
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K @ K)

    return R

def detect_ideal_gravity(g_sensor=None):
    '''
    Detects the ideal direction of the global gravity axis, accounting for sign.
    The input must be a (3,) np.array containing a measure of the still accelerometer you want to align.
    
    Inputs:
        - g_sensor (np.array (3,)): the direction of the average acceleromter when the person is standing still

    Returns:
        - g_global (np.array (3,)): the direction of ideal gravity
    '''
    
    g_global = np.zeros(3) # Initialize g_global
    g_unit = unit_vector(g_sensor) 

    # Find where the most prominent axis is, irrespective of the sign for now
    main_axis_idx = np.argmax(np.abs(g_unit)) 

    # Replace that index in g_global with 1, accounting for sign
    g_global[main_axis_idx] = 1 * np.sign(g_unit[main_axis_idx])
    return g_global

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