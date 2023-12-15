# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:45:25 2021

@author: amarmore

## Author : Axel Marmoret, based on Florian Voorwinden's code during its internship.

"""

import numpy as np
import time
import tensorly as tl
import nn_fac.utils.errors as err

def switch_alternate_mu(data, U, V, beta, matrix):
    """
    Encapsulates the switch between the two multiplicative update rules.
    """
    if matrix in ["U", "W"]:
        return mu_betadivmin(U, V, data, beta)
    elif matrix in ["V", "H"]:
        return np.transpose(mu_betadivmin(V.T, U.T, data.T, beta))
    else:
        raise err.InvalidArgumentValue(f"Invalid value for matrix: got {matrix}, but it must be 'U' or 'W' for the first matrix, and 'V' or 'H' for the second one.") from None

def mu_betadivmin(U, V, M, beta):
    """
    =====================================================
    Beta-Divergence NMF solved with Multiplicative Update
    =====================================================

    Computes an approximate solution of a beta-NMF
    [3] with the Multiplicative Update rule [2,3].
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.

    Conversely than in [1], the NNLS problem is solved for the beta-divergence,
    as studied in [3]:

            min_{U >= 0} beta_div(M, UV)

    The update rule of this algorithm is defined in [3].

    Parameters
    ----------
    U : m-by-r array
        The first factor of the NNLS, the one which will be updated.
    V : r-by-n array
        The second factor of the NNLS, which won't be updated.
    M : m-by-n array
        The initial matrix, to approach.
    beta : Nonnegative float
        The beta coefficient for the beta-divergence.

    Returns
    -------
    U: array
        a m-by-r nonnegative matrix \approx argmin_{U >= 0} beta_div(M, UV)

    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [2] D. Lee and H. S. Seung, Learning the parts of objects by non-negative
    matrix factorization., Nature, vol. 401, no. 6755, pp. 788–791, 1999.

    [3] C. Févotte and J. Idier, Algorithms for nonnegative matrix
    factorization with the beta-divergence, Neural Computation,
    vol. 23, no. 9, pp. 2421–2456, 2011.
    """

    if beta < 0:
        raise err.InvalidArgumentValue("Invalid value for beta: negative one.") from None

    epsilon = 1e-12

    K = np.dot(U,V)

    if beta == 1:
        K_inverted = K**(-1)
        line = np.sum(V.T,axis=0)
        denom = np.array([line for i in range(np.shape(K)[0])])
        return np.maximum(U * (np.dot((K_inverted*M),V.T) / denom),epsilon)
    elif beta == 2:
        denom = np.dot(K,V.T)
        return np.maximum(U * (np.dot(M,V.T) / denom), epsilon)
    elif beta == 3:
        denom = np.dot(K**2,V.T)
        return np.maximum(U * (np.dot((K * M),V.T) / denom) ** gamma(beta), epsilon)
    else:
        denom = np.dot(K**(beta-1),V.T)
        return np.maximum(U * (np.dot((K**(beta-2) * M),V.T) / denom) ** gamma(beta), epsilon)

def mu_tensorial(G, factors, tensor, beta):
    """
    This function is used to update the core G of a
    nonnegative Tucker Decomposition (NTD) [1] with beta-divergence [3]
    and Multiplicative Updates [2].

    See ntd.py of this module for more details on the NTD (or [1])

    TODO: expand this docstring.

    Parameters
    ----------
    G : tensorly tensor
        Core tensor at this iteration.
    factors : list of tensorly tensors
        Factors for NTD at this iteration.
    T : tensorly tensor
        The tensor to estimate with NTD.
    beta : Nonnegative float
        The beta coefficient for the beta-divergence.

    Returns
    -------
    G : tensorly tensor
        Update core in NTD.

    References
    ----------
    [1] Tamara G Kolda and Brett W Bader. "Tensor decompositions and applications",
    SIAM review 51.3 (2009), pp. 455{500.

    [2] D. Lee and H. S. Seung, Learning the parts of objects by non-negative
    matrix factorization., Nature, vol. 401, no. 6755, pp. 788–791, 1999.

    [3] C. Févotte and J. Idier, Algorithms for nonnegative matrix
    factorization with the beta-divergence, Neural Computation,
    vol. 23, no. 9, pp. 2421–2456, 2011.
    """

    if beta < 0:
        raise err.InvalidArgumentValue("Invalid value for beta: negative one.") from None

    epsilon = 1e-12
    K = tl.tenalg.multi_mode_dot(G,factors)

    if beta == 1:
        L1 = np.ones(np.shape(K))
        L2 = K**(-1) * tensor

    elif beta == 2:
        L1 = K
        L2 = np.ones(np.shape(K)) * tensor

    elif beta == 3:
        L1 = K**2
        L2 = K * tensor

    else:
        L1 = K**(beta-1)
        L2 = K**(beta-2) * tensor

    return np.maximum(G * (tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors]) / tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors])) ** gamma(beta) , epsilon)

def gamma(beta):
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
