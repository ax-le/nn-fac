# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:52:21 2019

@author: amarmore
"""

import numpy as np
import time
import nn_fac.nnls as nnls
import tensorly as tl
from nimfa.methods import seeding

def ntf(tensor, rank, init = "random", factors_0 = [], n_iter_max=100, tol=1e-8,
           sparsity_coefficients = [], fixed_modes = [], normalize = [],
           verbose=False, return_errors=False):

    """
    ======================================
    Nonnegative Tensor Factorization (NTF)
    ======================================

    Factorization of a tensor T in nonnegative matrices,
    corresponding to the PARAFAC decomposition of this tensor

    Precisely, each matrix corresponds the concatenation of all column vectors
    coming from PARAFAC decomposition on a tensor mode

    Hence, this results in as much factors as tensor modes

    All of this matrices are of size I_n*rank, with I_n the size of the n-th tensor mode

    The factorization problem is solved by fixing alternatively
    all factors except one, and resolving the problem on this one.
    Factorwise, the problem is reduced to a Nonnegative Least Squares problem.

    The optimization problem, for M_n, the n-th factor, is:

        min_{M_n}||T - M_n (khatri-rao(M_l))^T||_Fro^2
        + 2 * sparsity_coefficients[n] * (\sum\limits_{j = 0}^{r}||V[k,:]||_1)

    With:

        l the index of all the modes except the n-th one
        ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
        ||a||_1 = \sum_{i} abs(a_{i}) (Elementwise L1 norm)

    More precisely, the chosen optimization algorithm is the HALS [1],
    which updates each factor columnwise, fixing every other columns.

    Parameters
    ----------
    tensor: nonnegative tensor
        The tensor T, which is factorized
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
            designed for NMF resolution. See [2] for details.
            This nndsvd if performed via the nimfa toolbox [3].
        - If set to custom:
            factors_0 (see below) will be used for the initialization
        Default: random
    factors_0: None or list of array of nonnegative floats
        A custom initialization of the factors, used only in "custom" init mode.
        Default: None
    n_iter_max: integer
        The maximal number of iteration before stopping the algorithm
        Default: 100
    tol: float
        Threshold on the improvement in reconstruction error.
        Between two iterations, if the reconstruction error difference is
        below this threshold, the algorithm stops.
        Default: 1e-8
    sparsity_coefficients: array of float (as much as the number of modes)
        The sparsity coefficients on U and V respectively.
        If set to None, the algorithm is computed without sparsity
        Default: [],
    fixed_modes: array of integers (between 0 and the number of modes)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: []
    normalize: array of boolean (as much as the number of modes)
        A boolean whereas the factors need to be normalized.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise for U, linewise for V)
        Default: []
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
    np.array(factors): numpy array
        An array containing all the factors computed with PARAFAC decomposition
    errors: list
        A list of reconstruction errors at each iteration of the algorithm.
    toc: list
        A list with accumulated time at each iterations

    Example
    -------
    >>> import numpy as np
    >>> import ntf
    >>> rank = 6
    >>> U_lines = 100
    >>> V_lines = 125
    >>> W_lines = 50
    >>> factors_0 = []
    >>> factors_0.append(np.random.rand(U_lines, rank))
    >>> factors_0.append(np.random.rand(V_lines, rank))
    >>> factors_0.append(np.random.rand(W_lines, rank))
    >>> T = ntf.tl.kruskal_tensor.kruskal_to_tensor(factors_0)
    >>> factors = ntf.ntf(T, rank, init = "random", n_iter_max = 500, tol = 1e-8,
               sparsity_coefficients = [None, None, None], fixed_modes = [], normalize = [False, False, False],
               verbose = True, return_errors = False)

    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [2]: Christos Boutsidis and Efstratios Gallopoulos. "SVD based
    initialization: A head start for nonnegative matrix factorization",
    Pattern Recognition 41.4 (2008), pp. 1350{1362.

    [3]: Blalz Zupan et al. "Nimfa: A python library for nonnegative matrix
    factorization", Journal of Machine Learning Research 13.Mar (2012),
    pp. 849{853.

    [4] J. Kossai et al. "TensorLy: Tensor Learning in Python",
    arxiv preprint (2018)

    - Tamara G Kolda and Brett W Bader. "Tensor decompositions and applications",
    SIAM review 51.3 (2009), pp. 455{500.

    - Jeremy E Cohen. "About notations in multiway array processing",
    arXiv preprint arXiv:1511.01306, (2015).
    """

    factors = []
    nb_modes = len(tensor.shape)

    if init.lower() == "random":
        for mode in range(nb_modes):
            factors.append(tl.tensor(np.random.rand(tensor.shape[mode], rank)))

    elif init.lower() == "nndsvd":
        for mode in range(nb_modes):
            if tensor.shape[mode] < rank:
                current_factor = np.random.rand(tensor.shape[mode], rank)
            else:
                current_factor, useless_variable = seeding.Nndsvd().initialize(tl.unfold(tensor, mode), rank, {'flag': 0})
            factors.append(tl.tensor(current_factor))

    elif init.lower() == "custom":
        factors = factors_0
        if len(factors) != nb_modes:
            raise Exception("Custom initialization, but not enough factors")
        else:
            for array in factors:
                if array is None:
                    raise Exception("Custom initialization, but one factor is set to 'None'")

    else:
        raise Exception('Initialization type not understood')

    return compute_ntf(tensor, rank, factors, n_iter_max=n_iter_max, tol=tol,
                       sparsity_coefficients = sparsity_coefficients, fixed_modes = fixed_modes, normalize = normalize,
                       verbose=verbose, return_errors=return_errors)

def compute_ntf(tensor_in, rank, factors_in, n_iter_max=100, tol=1e-8,
           sparsity_coefficients = [], fixed_modes = [], normalize = [],
           verbose=False, return_errors=False):

    """
    Computation of a Nonnegative matrix factorization via
    hierarchical alternating least squares (HALS) [1],
    with factors_in as initialization.

    Parameters
    ----------
    tensor_in: nonnegative tensor
        The tensor T, which is factorized
    rank: integer
        The rank of the decomposition
    factors_in: list of array of nonnegative floats
        The initial factors
    n_iter_max: integer
        The maximal number of iteration before stopping the algorithm
        Default: 100
    tol: float
        Threshold on the improvement in reconstruction error.
        Between two iterations, if the reconstruction error difference is
        below this threshold, the algorithm stops.
        Default: 1e-8
    sparsity_coefficients: List of float (as much as the number of modes)
        The sparsity coefficients on U and V respectively.
        If set to None, the algorithm is computed without sparsity
        Default: [],
    fixed_modes: List of integers (between 0 and the number of modes)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: []
    normalize: List of boolean (as much as the number of modes)
        A boolean whereas the factors need to be normalized.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise)
        Default: []
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
    np.array(factors): numpy array
        An array containing all the factors computed with PARAFAC decomposition
    errors: list
        A list of reconstruction errors at each iteration of the algorithm.
    toc: list
        A list with accumulated time at each iterations

    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    - Tamara G Kolda and Brett W Bader. "Tensor decompositions and applications",
    SIAM review 51.3 (2009), pp. 455{500.

    - Jeremy E Cohen. "About notations in multiway array processing",
    arXiv preprint arXiv:1511.01306, (2015).
    """

    # initialisation - store the input varaibles
    factors = factors_in.copy()
    tensor = tensor_in.copy()
    norm_tensor = tl.norm(tensor, 2)

    # set init if problem
    nb_modes = len(tensor.shape)
    if sparsity_coefficients == None or len(sparsity_coefficients) != nb_modes:
        print("Irrelevant number of sparsity coefficient (different from the number of modes), they have been set to None.")
        sparsity_coefficients = [None for i in range(nb_modes)]
    if fixed_modes == None:
        fixed_modes = []
    if normalize == None or len(normalize) != nb_modes:
        print("Irrelevant number of normalization booleans (different from the number of modes), they have been set to False.")
        normalize = [False for i in range(nb_modes)]

    # initialisation - declare local varaibles
    rec_errors = []
    tic = time.time()
    toc = []

    # initialisation - unfold the tensor according to the modes
    unfolded_tensors = []
    for mode in range(tl.ndim(tensor)):
        unfolded_tensors.append(tl.base.unfold(tensor, mode))

    # Iterate over one step of NTF
    for iteration in range(n_iter_max):
        # One pass of least squares on each updated mode
        factors, rec_error = one_ntf_step(unfolded_tensors, rank, factors, norm_tensor,
                                          sparsity_coefficients, fixed_modes, normalize)
        # Store the computation time
        toc.append(time.time() - tic)

        rec_errors.append(rec_error)

        if verbose:
            if iteration == 0:
                print('reconstruction error={}'.format(rec_errors[iteration]))
            else:
                if rec_errors[-2] - rec_errors[-1] > 0:
                    print('reconstruction error={}, variation={}.'.format(
                            rec_errors[-1], rec_errors[-2] - rec_errors[-1]))
                else:
                    print('\033[91m' + 'reconstruction error={}, variation={}.'.format(
                            rec_errors[-1], rec_errors[-2] - rec_errors[-1]) + '\033[0m')

        if iteration > 0 and abs(rec_errors[-2] - rec_errors[-1]) < tol:
            # Stop condition: relative error between last two iterations < tol
            if verbose:
                print('converged in {} iterations.'.format(iteration))
            break


    if return_errors:
        return np.array(factors), rec_errors, toc
    else:
        return np.array(factors)


def one_ntf_step(unfolded_tensors, rank, in_factors, norm_tensor,
                 sparsity_coefficients, fixed_modes, normalize,
                 alpha=0.5, delta=0.01):
    """
    One pass of Hierarchical Alternating Least Squares update along all modes

    Update the factors by solving a least squares problem per mode, as described in [1].

    Note that the unfolding order is the one described in [2], which is different from [1].

    This function is strictly superior to a least squares solver ran on the
    matricized problems min_X ||Y - AX||_F^2 since A is structured as a
    Kronecker product of other factors.

    Parameters
    ----------
    unfolded_tensors: list of array
        The spectrogram tensor, unfolded according to all its modes.
    in_factors: list of array
        Current estimates for the PARAFAC decomposition of
        tensor. The value of factor[update_mode]
        will be updated using a least squares update.
        The values in in_factors are not modified.
    rank: int
        Rank of the decomposition.
    norm_tensor : float
        The Frobenius norm of the input tensor
    sparsity_coefficients : List of floats
        sparsity coefficients for every mode.
    fixed_modes : List of integers
        Indexes of modes that are not updated
    normalize: List of boolean (as much as the number of modes)
        A boolean where the factors need to be normalized.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise)
    alpha : positive float
        Ratio between outer computations and inner loops. Typically set to
        0.5 or 1.
        Default: 0.5
    delta : float in [0,1]
        Early stop criterion, while err_k > delta*err_0. Set small for
        almost exact nnls solution, or larger (e.g. 1e-2) for inner loops
        of a PARAFAC computation.
        Default: 0.01


    Returns
    -------
    np.array(factors): numpy array
        An array containing all the factors computed with PARAFAC decomposition
    errors: list
        A list of reconstruction errors at each iteration of the algorithm.
    toc: list
        A list with accumulated time at each iterations

    References
    ----------
    [1] Tamara G Kolda and Brett W Bader. "Tensor decompositions and applications",
    SIAM review 51.3 (2009), pp. 455{500.

    [2] Jeremy E Cohen. "About notations in multiway array processing",
    arXiv preprint arXiv:1511.01306, (2015).
    """

    # Avoiding errors
    for fixed_value in fixed_modes:
        sparsity_coefficients[fixed_value] = None

    # Copy
    factors = in_factors.copy()

    # Generating the mode update sequence
    gen = [mode for mode in range(len(unfolded_tensors)) if mode not in fixed_modes]

    for mode in gen:

        tic = time.time()

        # Computing Hadamard of cross-products
        cross = tl.tensor(tl.ones((rank,rank)))#, **tl.context(tensor))
        for i, factor in enumerate(factors):
            if i != mode:
                cross *= tl.dot(tl.transpose(factor),factor)

        # Computing the Khatri Rao product
        krao = tl.tenalg.khatri_rao(factors, skip_matrix = mode)
        rhs = tl.dot(unfolded_tensors[mode],krao)

        timer = time.time() - tic

        # Call the hals resolution with nnls, optimizing the current mode
        factors[mode] = tl.transpose(nnls.hals_nnls_acc(tl.transpose(rhs), cross, tl.transpose(factors[mode]),
               maxiter=100, atime=timer, alpha=alpha, delta=delta,
               sparsity_coefficient = sparsity_coefficients[mode], normalize = normalize[mode])[0])

    # Adding the l1 norm value to the reconstruction error
    sparsity_error = 0
    for index, sparse in enumerate(sparsity_coefficients):
        if sparse:
            sparsity_error += 2 * (sparse * np.linalg.norm(factors[index], ord=1))

    # error computation (improved using precomputed quantities)
    rec_error = norm_tensor ** 2 - 2*tl.dot(tl.tensor_to_vec(factors[mode]),tl.tensor_to_vec(rhs)) +  tl.norm(tl.dot(factors[mode],tl.transpose(krao)),2)**2
    rec_error = (rec_error ** (1/2) + sparsity_error) / norm_tensor

    return factors, rec_error
