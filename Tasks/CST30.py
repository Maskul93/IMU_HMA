import numpy as np
import pandas as pd
from SignalProcessing.Filters import besfilt

def get_transition_timings(acc=None, fs=None, cc=0, ap=2, loc='thigh'):
    '''
    Finds the zero crossings from the jerk of AP and CC axes of thigh-worn accelerometer.

    Inputs:
        - acc (np.ndarray (N, 3)): the thigh accelerometer recordings of the 30CST
        - fs (float): sampling frequency of the acceleromter
        - cc (int): column of `acc` where the CC acceleration is stored
        - ap (int): column of `acc` where the AP acceleration is stored
    
    Returns:
        - pd.DataFrame: a DataFrame containing two columns representing start and end of each detected transition
    '''

    j_cc = np.gradient(acc[:, cc], 1/fs)
    j_ap = np.gradient(acc[:, ap], 1/fs)

    if loc == 'thigh':
        j = besfilt(j_cc + j_ap, 4, 1, fs)
    elif loc == 'chest':
        j = besfilt(j_cc, 4, 1, fs)

    # find zero-crossings
    zc = np.where(np.diff(np.sign(j)))[0]
    transitions_list = [(zc[i], zc[i+1]-1) for i in range(len(zc)-1)]

    return pd.DataFrame(transitions_list, columns=['sample_i', 'sample_f'])

def sigmoid_template(x):
    '''
    Creates a sigmoid placed on the given time-axis. Required for characterizing transitions.
    
    Inputs:
        - x (np.array (N,)): the equally spaced time-array where the sigmoid is created

    Returns:
        - np.array (N, ): the numerical sigmoid function
    '''
    return 1/(1 + np.exp(-x))

def characterize_transition(acc=None, sample_i=None, sample_f=None):
    '''
    Detects whether a transition is a Sit-to-Stand or a Stand-to-Sit based on the sigmoid similarity method.
    
    Inputs:
        - acc (np.ndarray (N, 3)): the thigh accelerometer recordings of the 30CST
        - sample_i (int): sample where the candidate transition starts
        - sample_f (int): sample where the candidate transition ends

    Returns:
        - pd.Series: Series containing the transition kind
    '''

    a_cc = acc[sample_i : sample_f, 0]
    a_ap = acc[sample_i : sample_f, 2]

    # Initialize sigmoid
    t = np.linspace(-5, 5, len(a_cc)) 
    sigm = sigmoid_template(t)
    
    # confidence analysis:
    c_cc = np.corrcoef(sigm, a_cc)[0][1]
    c_ap = np.corrcoef(sigm, a_ap)[0][1]

    if all(c > 0.5 for c in [c_cc, c_ap]):
        kind = 'ST-SI'
    elif all(c < -0.5 for c in [c_cc, c_ap]):
        kind = 'SI-ST'
    else:
        kind = np.nan

    return pd.Series({'kind': kind})