# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:45:25 2021

@author: amarmore    

## Author : Axel Marmoret, based on Florian Voorwinden's code during its internship.

"""

import numpy as np
import nn_fac.errors as err

def beta_divergence(a, b, beta):
    """
    Compute the beta-divergence of two floats or arrays a and b,
    as defined in [3].

    Parameters
    ----------
    a : float or array
        First argument for the beta-divergence.
    b : float or array
        Second argument for the beta-divergence. 
    beta : float
        the beta factor of the beta-divergence.
    
    Returns
    -------
    float
        Beta-divergence of a and b.
        
    References
    ----------
    [1] C. Févotte and J. Idier, Algorithms for nonnegative matrix 
    factorization with the beta-divergence, Neural Computation, 
    vol. 23, no. 9, pp. 2421–2456, 2011.
    """
    if beta < 0:
        raise err.InvalidArgumentValue("Invalid value for beta: negative one.") from None
    
    if beta == 1:
        return np.sum(a * np.log(a/b, where=(a!=0)) - a + b)
    elif beta == 0:
        return np.sum(a/b - np.log(a/b, where=(a!=0)) - 1)
    else:
        return np.sum(1/(beta*(beta -1)) * (a**beta + (beta - 1) * b**beta - beta * a * (b**(beta-1))))