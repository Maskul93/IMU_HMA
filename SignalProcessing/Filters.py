import numpy as np
from scipy.signal import bessel, butter, filtfilt

def bwfilt(sig=None, order=None, fc=None, fs=None, btype='low'):
    '''
    Easier implementation of the Butterworth filter from scipy. works only on a single axis per time.
    Input: 
        - sig (np.array): the signal to be filtered. 
        - order (int): filter order.
        - fc (float): cut-off frequency.
        - fs (float): sampling frequency of the input signal.

    Returns:
        - np.array: the filtered signal.
    '''
    b, a = butter(order, fc / (.5 * fs), btype=btype, analog=False)
    
    if sig.ndim == 1:
        return filtfilt(b,a,sig)
    else:
        for i in range(sig.shape[-1]):
            sig[:, i] = filtfilt(b,a,sig[:, i])
            
        return sig

def besfilt(sig=None, order=None, fc=None, fs=None, btype='low'):
    '''
    Easier implementation of the Bessel filter from scipy. works only on a single axis per time.
    Input: 
        - sig (np.array): the signal to be filtered. 
        - order (int): filter order.
        - fc (float): cut-off frequency.
        - fs (float): sampling frequency of the input signal.

    Returns:
        - np.array: the filtered signal.
    '''
    b, a = bessel(order, fc / (.5 * fs), btype=btype, analog=False)
    
    if sig.ndim == 1:
        return filtfilt(b,a,sig)
    else:
        for i in range(sig.shape[-1]):
            sig[:, i] = filtfilt(b,a,sig[:, i])
            
        return sig