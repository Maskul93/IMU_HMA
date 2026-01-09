import numpy as np
from scipy.integrate import cumulative_trapezoid

def integrate_ndim(y_prime=None, fs=None):
    '''
    Integrates a signal using trapezoidal method, accounting for dimensionality.

    Inputs:
        - y_prime (np.ndarray (N,M)): the derivative function to be integrated
        - fs (float): the sampling frequency of the signal

    Returns:
        - y (np.ndarray (N,M)): the integrated function
    '''
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

def differentiate_ndim(y=None, fs=None):
    '''
    Differentiates a signal using np.gradient function, accounting for dimensionality.
    
    Inputs:
        - y (np.ndarray (N,M)): the primitive function to be differentiated
        - fs (float): the sampling frequency of the signal

    Returns:
        - y_prime (np.ndarray (N,M)): the derivative function
    '''
    y_prime = np.zeros_like(y)

    if y.ndim == 1: # Handling unidimensional array
        y_prime = np.gradient(y, 1/fs)
    elif y.ndim == 2: # Handling bidimensional array (e.g. [N x 3])
        for i in range(y.shape[1]): 
            y_prime[:, i] = np.gradient(y[:, i], 1/fs)
    else:
        print('More than two dimensions are still not handled. Returning an empty array')
        y_prime = np.empty(0)
    
    return y_prime