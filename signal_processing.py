import numpy as np
from scipy.signal import butter, bessel, filtfilt
from scipy.integrate import cumulative_trapezoid

def bwfilt(sig=None, order=None, fc=None, fs=None, btype='low'):
    # Initialize Butterworth filter with given parameters    
    b, a = butter(N=order, Wn=fc/(0.5*fs), btype=btype)

    # Initialize filtered signal
    sig_f = np.zeros_like(sig)

    if sig.ndim == 1: # Handling unidimensional array
        sig_f = filtfilt(b, a, sig)
    elif sig.ndim == 2: # Handling bidimensional array (e.g. [N x 3])
        for i in range(sig.shape[1]):
            sig_f[:, i] = filtfilt(b, a, sig[:, i])
    else:
        print('More than two dimensions are still not handled. Returning an empty array')
        sig_f = np.empty(0)
    
    return sig_f

def besfilt(sig=None, order=None, fc=None, fs=None, btype='low'):
    # Initialize Butterworth filter with given parameters    
    b, a = bessel(N=order, Wn=fc/(0.5*fs), btype=btype)

    # Initialize filtered signal
    sig_f = np.zeros_like(sig)

    if sig.ndim == 1: # Handling unidimensional array
        sig_f = filtfilt(b, a, sig)
    elif sig.ndim == 2: # Handling bidimensional array (e.g. [N x 3])
        for i in range(sig.shape[1]):
            sig_f[:, i] = filtfilt(b, a, sig[:, i])
    else:
        print('More than two dimensions are still not handled. Returning an empty array')
        sig_f = np.empty(0)
    
    return sig_f

def integrate_ndim(y_prime=None, fs=None):
    # Initialize time
    t = np.linspace(0, len(y_prime)/fs, len(y_prime))

    if y_prime.ndim == 1: # Handling unidimensional array
        y = cumulative_trapezoid(y=y_prime, x=t, initial=0)
    elif y_prime.ndim == 2:
        y = np.zeros_like(y_prime) # Handling bidimensional array (e.g. [N x 3])
        for i in range(y_prime.shape[1]):
            y[:, i] = cumulative_trapezoid(y=y_prime[:, i], x=t, initial=0)
    else:
        print('More than two dimensions are still not handled. Returning an empty array')
        y = np.empty(0)
    
    return y

def differentiate_ndmi(y=None, fs=None):
    '''Differentiate a signal using np.gradient function, accounting for dimensionality.'''
    y_prime = np.zeros_like(y)

    if y.ndim == 1:
        y_prime = np.gradient(y, 1/fs)
    elif y.ndim == 2:
        for i in range(y.shape[1]):
            y_prime[:, i] = np.gradient(y[:, i], 1/fs)
    else:
        print('More than two dimensions are still not handled. Returning an empty array')
        y_prime = np.empty(0)
    
    return y_prime
