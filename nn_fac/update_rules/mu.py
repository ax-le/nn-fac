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
from nn_fac.utils.beta_divergence import gamma_beta
import nn_fac.utils.normalize_wh as normalize_wh

epsilon = 1e-12

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
        return np.maximum(U * (np.dot((K * M),V.T) / denom) ** gamma_beta(beta), epsilon)
    else:
        denom = np.dot(K**(beta-1),V.T)
        return np.maximum(U * (np.dot((K**(beta-2) * M),V.T) / denom) ** gamma_beta(beta), epsilon)

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

    return np.maximum(G * (tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors]) / tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors])) ** gamma_beta(beta) , epsilon)

def simplex_proj_mu(data, W, H, beta, tol_update_lagrangian = 1e-6):
    # Projects H on the unit simplex, comes from 'Leplat, V., Gillis, N., & Idier, J. (2021). Multiplicative updates for NMF with β-divergences under disjoint equality constraints. SIAM Journal on Matrix Analysis and Applications, 42(2), 730-752. arXiv:2010.16223.'
    k,n = H.shape
    Jk1 = np.ones((k, 1))
    C=(W.T@(((W@H)**(beta-2)) * data))
    D=W.T@((W@H)**(beta-1))

    lagrangian_multipliers_0 = ((D[0,:] - (C[0,:] * H[0,:]))**(gamma_beta(beta))).T # np.zeros((n, 1))
    lagrangian_multipliers_0 = lagrangian_multipliers_0.reshape((n,1))
    lagrangian_multipliers = normalize_wh.update_lagragian_multipliers_simplex_projection(C, D, H, beta, lagrangian_multipliers_0, tol = tol_update_lagrangian, n_iter_max = 100)
    
    H = H * (C/((D-Jk1@lagrangian_multipliers.T)+epsilon))**(gamma_beta(beta))
    H = np.maximum(H,epsilon)

    return H
