# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:45:25 2021

@author: amarmore    

## Author : Axel Marmoret, based on Florian Voorwinden's code during its internship.

"""
import tensorly as tl
import nn_fac.utils.errors as err
import nn_fac.utils.tensorly_additional_utils as tl_additional_utils

def kl_divergence(a, b):
    return beta_divergence(a, b, beta=1)

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
        a_div_b = tl.where(b!=0, a/b, a/1e-12)
        return tl.sum(a * tl_additional_utils.log(a_div_b) - a + b)
    elif beta == 0:
        a_div_b = tl.where(b!=0, a/b, a/1e-12)
        return tl.sum(a_div_b - tl_additional_utils.log(a_div_b) - 1)
    else:
        return tl.sum(1/(beta*(beta -1)) * (a**beta + (beta - 1) * b**beta - beta * a * (b**(beta-1))))
    
def gamma_beta(beta):
    """
    Exponent of Fevotte and Idier [1], which guarantees the MU updates decrease the cost.
    
    See [1] for details.
    
    Parameters
    ----------
    beta : Nonnegative float
        The beta coefficient for the beta-divergence.

    Returns
    -------
    int : the exponent value
    
    References
    ----------
    [1]  C. Févotte and J. Idier, Algorithms for nonnegative matrix
    factorization with the beta-divergence, Neural Computation,
    vol. 23, no. 9, pp. 2421–2456, 2011.
    """
    if beta<1:
        return 1/(2-beta)
    if beta>2:
        return  1/(beta-1)
    else:
        return 1
