from numpy import exp, log
import numpy as np
from scipy.special import gammaln

def lfact(x):
    '''
        Compute the log factorial (log(x!)) using the scipy gammaln function

        Referred to as Stirlings approximation/formula for factorials.
    '''
    return gammaln(x + 1)

def nanexp(x):
    '''
        Compute the exponential of x (x:array_like), and replaces nan and inf with 

        Returns an array or scalar replacing Not a Number (NaN) with zero, (positive) infinity with a very large number and negative
        infinity with a very small (or negative) number.
    '''
    return np.nan_to_num(np.exp(x))
    