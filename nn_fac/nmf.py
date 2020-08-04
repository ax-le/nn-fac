# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:49:25 2019

@author: amarmore
"""

import numpy as np
import time
import nn_fac.nnls as nnls
from nimfa.methods import seeding


def nmf(data, rank, init = "random", U_0 = None, V_0 = None, n_iter_max=100, tol=1e-8,
           sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False],
           verbose=False, return_errors=False):

    """
    ======================================
    Nonnegative Matrix Factorization (NMF)
    ======================================

    Factorization of a matrix M in two nonnegative matrices U and V,
    such that the product UV approximates M.
    If M is of size m*n, U and V are resepctively of size m*r and r*n,
    r being the rank of the decomposition (parameter)
    Typically, this method is used as a dimensionality reduction technique,
    or for source separation.

    The objective function is:

        ||M - UV||_Fro^2
        + sparsity_coefficients[0] * (\sum\limits_{j = 0}^{r}||U[:,k]||_1)
        + sparsity_coefficients[1] * (\sum\limits_{j = 0}^{r}||V[k,:]||_1)

    With:

        ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
        ||a||_1 = \sum_{i} abs(a_{i}) (Elementwise L1 norm)

    The objective function is minimized by fixing alternatively
    one of both factors U and V and optimizing on the other one,
    the problem being reduced to a Nonnegative Least Squares problem.

    More precisely, the chosen optimization algorithm is the HALS [1],
    which updates each factor columnwise, fixing every other columns.

    Parameters
    ----------
    data: nonnegative array
        The matrix M, which is factorized
    rank: integer
        The rank of the decomposition
    init: "random" | "nndsvd" | "custom" |
        - If set to random:
            Initialize with random factors of the correct size.
            The randomization is the uniform distribution in [0,1),
            which is the default from numpy random.
        - If set to nnsvd:
            Corresponds to a Nonnegative Double Singular Value Decomposition
            (NNDSVD) initialization, which is a data based initialization,
            designed for NMF. See [2] for details.
            This NNDSVD if performed via the nimfa toolbox [3].
        - If set to custom:
            U_0 and V_0 (see below) will be used for the initialization
        Default: random
    U_0: None or array of nonnegative floats
        A custom initialization of U, used only in "custom" init mode.
        Default: None
    V_0: None or array of nonnegative floats
        A custom initialization of V, used only in "custom" init mode.
        Default: None
    n_iter_max: integer
        The maximal number of iteration before stopping the algorithm
        Default: 100
    tol: float
        Threshold on the improvement in reconstruction error.
        Between two iterations, if the reconstruction error difference is
        below this threshold, the algorithm stops.
        Default: 1e-8
    sparsity_coefficients: List of float (two)
        The sparsity coefficients on U and V respectively.
        If set to None, the algorithm is computed without sparsity
        Default: [None, None],
    fixed_modes: List of integers (between 0 and 2)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: []
    normalize: List of boolean (two)
        A boolean whereas the factors need to be normalized.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise for U, linewise for V)
        Default: [False, False]
    verbose: boolean
        Indicates whether the algorithm prints the successive
        reconstruction errors or not
        Default: False
    return_errors: boolean
        Indicates whether the algorithm should return all reconstruction errors
        and computation time of each iteration or not
        Default: False

    Returns
    -------
    U, V: numpy arrays
        Factors of the NMF
    errors: list
        A list of reconstruction errors at each iteration of the algorithm.
    toc: list
        A list with accumulated time at each iterations

    Example
    -------
    >>> import numpy as np
    >>> import nmf
    >>> rank = 5
    >>> U_lines = 100
    >>> V_col = 125
    >>> U_0 = np.random.rand(U_lines, rank)
    >>> V_0 = np.random.rand(rank, V_col)
    >>> M = U_0@V_0
    >>> U, V = nmf.nmf(M, rank, init = "random", n_iter_max = 500, tol = 1e-8,
               sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False],
               verbose=True, return_errors = False)

    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [2]: C. Boutsidis and E. Gallopoulos. "SVD based
    initialization: A head start for nonnegative matrix factorization,"
    Pattern Recognition 41.4 (2008), pp. 1350{1362.

    [3]: B. Zupan et al. "Nimfa: A python library for nonnegative matrix
    factorization", Journal of Machine Learning Research 13.Mar (2012),
    pp. 849{853.
    """

    if init.lower() == "random":
        k, n = data.shape
        U_0 = np.random.rand(k, rank)
        V_0 = np.random.rand(rank, n)

    elif init.lower() == "nndsvd":
        U_0, V_0 = seeding.Nndsvd().initialize(data, rank, {'flag': 0})

    elif init.lower() == "custom":
        if U_0 is None or V_0 is None:
            raise Exception("Custom initialization, but one factor is set to 'None'")

    else:
        raise Exception('Initialization type not understood')

    return compute_nmf(data, rank, U_0, V_0, n_iter_max=n_iter_max, tol=tol,
                   sparsity_coefficients = sparsity_coefficients, fixed_modes = fixed_modes, normalize = normalize,
                   verbose=verbose, return_errors=return_errors)




# Author : Jeremy Cohen, modified by Axel Marmoret
def compute_nmf(data, rank, U_in, V_in, n_iter_max=100, tol=1e-8,
           sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False],
           verbose=False, return_errors=False):
    """
    Computation of a Nonnegative matrix factorization via
    hierarchical alternating least squares (HALS) [1],
    with U_in and V_in as initialization.

    Parameters
    ----------
    data: nonnegative array
        The matrix M, which is factorized, of size m*n
    rank: integer
        The rank of the decomposition
    U_in: array of floats
        Initial U factor, of size m*r
    V_in: array of floats
        Initial V factor, of size r*n
    n_iter_max: integer
        The maximal number of iteration before stopping the algorithm
        Default: 100
    tol: float
        Threshold on the improvement in reconstruction error.
        Between two iterations, if the reconstruction error difference is
        below this threshold, the algorithm stops.
        Default: 1e-8
    sparsity_coefficients: List of float (two)
        The sparsity coefficients on U and V respectively.
        If set to None, the algorithm is computed without sparsity
        Default: [None, None],
    fixed_modes: List of integers (between 0 and 2)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: []
    normalize: List of boolean (two)
        A boolean whereas the factors need to be normalized.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise for U, linewise for V)
        Default: [False, False]
    verbose: boolean
        Indicates whether the algorithm prints the successive
        reconstruction errors or not
        Default: False
    return_errors: boolean
        Indicates whether the algorithm should return all reconstruction errors
        and computation time of each iteration or not
        Default: False

    Returns
    -------
    U, V: numpy arrays
        Factors of the NMF
    errors: list
        A list of reconstruction errors at each iteration of the algorithm.
    toc: list
        A list with accumulated time at each iterations

    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.
    """

    # initialisation
    U = U_in.copy()
    V = V_in.copy()
    rec_errors = []
    norm_data = np.linalg.norm(data)
    tic = time.time()
    toc = []

    if sparsity_coefficients == None:
        sparsity_coefficients = [None, None]
    if fixed_modes == None:
        fixed_modes = []
    if normalize == None or normalize == False:
        normalize = [False, False]

    for iteration in range(n_iter_max):

        # One pass of least squares on each updated mode
        U, V, rec_error = one_nmf_step(data, rank, U, V, norm_data,
                                       sparsity_coefficients, fixed_modes, normalize)

        toc.append(time.time() - tic)
        if tol:
            rec_errors.append(rec_error)

            if verbose:
                if iteration == 0:
                    print('reconstruction error={}'.format(rec_errors[iteration]))
                else:
                    if rec_errors[-2] - rec_errors[-1] > 0:
                        print('reconstruction error={}, variation={}.'.format(
                                rec_errors[-1], rec_errors[-2] - rec_errors[-1]))
                    else:
                        # print in red when the reconstruction error is negative (shouldn't happen)
                        print('\033[91m' + 'reconstruction error={}, variation={}.'.format(
                                rec_errors[-1], rec_errors[-2] - rec_errors[-1]) + '\033[0m')

            if iteration > 0 and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                # Stop condition: relative error between last two iterations < tol
                if verbose:
                    print('converged in {} iterations.'.format(iteration))
                break

    if return_errors:
        return np.array(U), np.array(V), rec_errors, toc
    else:
        return np.array(U), np.array(V)


def one_nmf_step(data, rank, U_in, V_in, norm_data,
                 sparsity_coefficients, fixed_modes, normalize):
    """
    One pass of updates for each factor in NMF
    Update the factors by solving a nonnegative least squares problem per mode.

    Parameters
    ----------
    data: nonnegative array
        The matrix M, which is factorized, of size m*n
    rank: integer
        The rank of the decomposition
    U_in: array of floats
        Initial U factor, of size m*r
    V_in: array of floats
        Initial V factor, of size r*n
    norm_data: float
        The Frobenius norm of the input matrix (data)
    sparsity_coefficients: List of float (two)
        The sparsity coefficients on U and V respectively.
        If set to None, the algorithm is computed without sparsity
        Default: [None, None],
    fixed_modes: List of integers (between 0 and 2)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: []
    normalize: List of boolean (two)
        A boolean whereas the factors need to be normalized.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise for U, linewise for V)
        Default: [False, False]

    Returns
    -------
    U, V: numpy arrays
        Factors of the NMF
    rec_error:
        The reconstruction error of this NMF step
    """

    if len(sparsity_coefficients) != 2:
        raise ValueError("NMF needs 2 sparsity coefficients to be performed")

    # Copy
    U = U_in.copy()
    V = V_in.copy()

    if 0 not in fixed_modes:
        # U update

        # Set timer for acceleration in hals_nnls_acc
        tic = time.time()

        # Computing cross products
        VVt = np.dot(V,np.transpose(V))
        VMt = np.dot(V,np.transpose(data))

        # End timer for acceleration in hals_nnls_acc
        timer = time.time() - tic

        # Compute HALS/NNLS resolution
        U = np.transpose(nnls.hals_nnls_acc(VMt, VVt, np.transpose(U_in), maxiter=100, atime=timer, alpha=0.5, delta=0.01,
                                            sparsity_coefficient = sparsity_coefficients[0], normalize = normalize[0], nonzero = False)[0])

    if 1 not in fixed_modes:
        # V update

        # Set timer for acceleration in hals_nnls_acc
        tic = time.time()

        # Computing cross products
        UtU = np.dot(np.transpose(U),U)
        UtM = np.dot(np.transpose(U),data)

        # End timer for acceleration in hals_nnls_acc
        timer = time.time() - tic

        # Compute HALS/NNLS resolution
        V = nnls.hals_nnls_acc(UtM, UtU, V_in, maxiter=100, atime=timer, alpha=0.5, delta=0.01,
                               sparsity_coefficient = sparsity_coefficients[1], normalize = normalize[1], nonzero = False)[0]

    sparsity_coefficients = np.where(np.array(sparsity_coefficients) == None, 0, sparsity_coefficients)

    rec_error = np.linalg.norm(data-np.dot(U,V), ord='fro') ** 2 + 2 * (sparsity_coefficients[0] * np.linalg.norm(U, ord=1) + sparsity_coefficients[1] * np.linalg.norm(V, ord=1))

    rec_error = rec_error/norm_data

    return U, V, rec_error
