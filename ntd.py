# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:52:21 2019

@author: amarmore
"""
import numpy as np
import scipy
import time
import nnls
import tensorly as tl
from tensorly.decomposition import tucker as tl_tucker
import math
import errors as err

# TODO: Normalization on the core tensor is only possible for the last mode (our use case).
# To make dependant of a parameter and available for every mode.

def ntd(tensor, ranks, init = "random", core_0 = None, factors_0 = [], n_iter_max=100, tol=1e-6,
           sparsity_coefficients = [], fixed_modes = [], normalize = [], hals = False,
           verbose=False, return_errors=False, deterministic=False):

    """
    ======================================
    Nonnegative Tucker Decomposition (NTD)
    ======================================

    Factorization of a tensor T in nonnegative matrices,
    linked by a nonnegative core tensor, of dimensions equal to the ranks
    (in general smaller than the tensor).
    See more details about the NTD in [1].

    For example, in the third-order case, resolution of:
        T \approx (W \otimes H \otimes Q) G
        
    In this example, W, H and Q are the factors, one per mode, and G is the core tensor.
    W is of size T.shape[0] * ranks[0],
    H is of size T.shape[1] * ranks[1],
    Q is of size T.shape[2] * ranks[2],
    G is of size ranks[0] * ranks[1] * ranks[2].

    More precisely, the chosen optimization algorithm is the HALS [2] for the factors,
    which updates each factor columnwise, fixing every other columns,
    and a projected gradient for the core,
    which reduces the memory neccesited to perfome HALS on the core.
    The projected gradient rule is derived by the authors, and doesn't appear in citation for now.
    
    Tensors are manipulated with the tensorly toolbox [3].
    
    In tensorly and in our convention, tensors are unfolded and treated as described in [4].

    Parameters
    ----------
    tensor: nonnegative tensor
        The tensor T, to factorize
    ranks: list of integers
        The ranks for each factor of the decomposition
    init: "random" | "tucker" | "custom" |
        - If set to random:
            Initializes with random factors of the correct size.
            The randomization is the uniform distribution in [0,1),
            which is the default from numpy random.
        - If set to tucker:
            Resolve a tucker decomposition of the tensor T and
            initializes the factors and the core as this resolution, clipped to be nonnegative.
            The tucker decomposition is performed with tensorly [3].
        - If set to custom:
            core_0 and factors_0 (see below) will be used for the initialization
        Default: random
    core_0: None or tensor of nonnegative floats
        A custom initialization of the core, used only in "custom" init mode.
        Default: None
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
    sparsity_coefficients: list of float (as much as the number of modes + 1 for the core)
        The sparsity coefficients on each factor and on the core respectively.
        If set to None or [], the algorithm is computed without sparsity
        Default: []
    fixed_modes: list of integers (between 0 and the number of modes + 1 for the core)
        Has to be set not to update a factor, taken in the order of modes and lastly on the core.
        Default: []
    normalize: list of boolean (as much as the number of modes + 1 for the core)
        A boolean whereas the factors need to be normalized.
        The normalization is a l_2 normalization on each of the rank components
        (For the factors, each column will be normalized, ie each atom of the dimension of the current rank).
        Default: []
    hals: boolean
        Whether to run hals (true) or gradient (false) update on the core.
        Default (and recommanded): false
    verbose: boolean
        Indicates whether the algorithm prints the successive reconstruction errors or not.
        Default: False
    return_errors: boolean
        Indicates whether the algorithm should return all reconstruction errors
        and computation time of each iteration or not.
        Default: False
    deterministic:
        Runs the algorithm as a deterministic way, by fixing seed in all possible randomisation.
        This is made to enhance reproducible research.

    Returns
    -------
    core: tensorly tensor
        The core tensor linking the factors of the decomposition
    factors: numpy #TODO: For tensorly pulling, replace numpy by backend
        An array containing all the factors computed with the NTD
    errors: list, only if return_errors == True
        A list of reconstruction errors at each iteration of the algorithm.
    toc: list, only if return_errors == True
        A list with accumulated time at each iterations

    Example
    -------
    TODO

    References
    ----------
    [1] Tamara G Kolda and Brett W Bader. "Tensor decompositions and applications",
    SIAM review 51.3 (2009), pp. 455{500.
    
    [2]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [3] J. Kossai et al. "TensorLy: Tensor Learning in Python",
    arxiv preprint (2018)

    [4] Jeremy E Cohen. "About notations in multiway array processing",
    arXiv preprint arXiv:1511.01306, (2015).
    """

    factors = []
    nb_modes = len(tensor.shape)
    
    if type(ranks) is int:
        ranks = [ranks for i in nb_modes]
    elif len(ranks) != nb_modes:
        raise err.InvalidRanksException("The number of ranks is different than the dim of the tensor, which is incorrect.") from None
    
    for i in range(nb_modes):
        if ranks[i] > tensor.shape[i]:
            ranks[i] = tensor.shape[i]
            #warnings.warn('The %s-th mode rank was larger than the shape of the tensor, which is incorrect. Set to the shape of the tensor' % i)
            print("The " + str(i) + "-th mode rank was larger than the shape of the tensor, which is incorrect. Set to the shape of the tensor")
    
    if init.lower() == "random":
        for mode in range(nb_modes):
            if deterministic:
                seed = np.random.RandomState(mode * 10)
                random_array = seed.rand(tensor.shape[mode], ranks[mode])
            else:
                random_array = np.random.rand(tensor.shape[mode], ranks[mode])
            factors.append(tl.tensor(random_array))
        if deterministic:
            seed = np.random.RandomState(nb_modes * 10)
            core = tl.tensor(seed.rand(np.prod(ranks)).reshape(tuple(ranks)))
        else:
            core = tl.tensor(np.random.rand(np.prod(ranks)).reshape(tuple(ranks)))

    elif init.lower() == "tucker":
        if deterministic:
            init_core, init_factors = tl_tucker(tensor, ranks, random_state = 8142)
        else:
            init_core, init_factors = tl_tucker(tensor, ranks)
        factors = [tl.abs(f) for f in init_factors]
        core = tl.abs(init_core)

    elif init.lower() == "chromas":
        if deterministic:
            init_core, init_factors = tl_tucker(tensor, ranks, random_state = 8142)
        else:
            init_core, init_factors = tl_tucker(tensor, ranks)
        init_factors[0] = np.identity(12) # En dure, à être modifié
        factors = [tl.abs(f) for f in init_factors]
        if 0 not in fixed_modes:
            fixed_modes.append(0)
        core = tl.abs(init_core)

    elif init.lower() == "custom":
        factors = factors_0
        core = core_0
        if len(factors) != nb_modes:
            raise err.CustomNotEngouhFactors("Custom initialization, but not enough factors")
        else:
            for array in factors:
                if array is None:
                    raise err.CustomNotValidFactors("Custom initialization, but one factor is set to 'None'")
            if core is None:
                raise err.CustomNotValidCore("Custom initialization, but the core is set to 'None'")

    else:
        raise err.InvalidInitializationType('Initialization type not understood: ' + init)

    return compute_ntd(tensor, ranks, core, factors, n_iter_max=n_iter_max, tol=tol,
                       sparsity_coefficients = sparsity_coefficients, fixed_modes = fixed_modes, normalize = normalize,
                       verbose=verbose, return_errors=return_errors, hals = hals, deterministic = deterministic)

def compute_ntd(tensor_in, ranks, core_in, factors_in, n_iter_max=100, tol=1e-6,
           sparsity_coefficients = [], fixed_modes = [], normalize = [], hals = False,
           verbose=False, return_errors=False, deterministic=False):

    """
    Computation of a Nonnegative Tucker Decomposition [1]
    via hierarchical alternating least squares (HALS) [2],
    with factors_in as initialization.
    
    Tensors are manipulated with the tensorly toolbox [3].
    
    In tensorly and in our convention, tensors are unfolded and treated as described in [4].

    Parameters
    ----------
    tensor_in: nonnegative tensor
        The tensor T, which is factorized
    rank: integer
        The rank of the decomposition
    core_in:
        The initial core
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
    sparsity_coefficients: list of float (as much as the number of modes + 1 for the core)
        The sparsity coefficients on each factor and on the core respectively.
        If set to None or [], the algorithm is computed without sparsity
        Default: []
    fixed_modes: list of integers (between 0 and the number of modes + 1 for the core)
        Has to be set not to update a factor, taken in the order of modes and lastly on the core.
        Default: []
    normalize: list of boolean (as much as the number of modes + 1 for the core)
        A boolean whereas the factors need to be normalized.
        The normalization is a l_2 normalization on each of the rank components
        (For the factors, each column will be normalized, ie each atom of the dimension of the current rank).
        Default: []
    hals: boolean
        Whether to run hals (true) or gradient (false) update on the core.
        Default (and recommanded): false
    verbose: boolean
        Indicates whether the algorithm prints the successive reconstruction errors or not.
        Default: False
    return_errors: boolean
        Indicates whether the algorithm should return all reconstruction errors
        and computation time of each iteration or not.
        Default: False
    deterministic:
        Runs the algorithm as a deterministic way, by fixing seed in all possible randomisation.
        This is made to enhance reproducible research.

    Returns
    -------
    core: tensorly tensor
        The core tensor linking the factors of the decomposition
    factors: numpy #TODO: For tensorly pulling, replace numpy by backend
        An array containing all the factors computed with the NTD
    errors: list, only if return_errors == True
        A list of reconstruction errors at each iteration of the algorithm.
    toc: list, only if return_errors == True
        A list with accumulated time at each iterations

    References
    ----------
    [1] Tamara G Kolda and Brett W Bader. "Tensor decompositions and applications",
    SIAM review 51.3 (2009), pp. 455{500.
    
    [2]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [3] J. Kossai et al. "TensorLy: Tensor Learning in Python",
    arxiv preprint (2018)

    [4] Jeremy E Cohen. "About notations in multiway array processing",
    arXiv preprint arXiv:1511.01306, (2015).
    """

    # initialisation - store the input varaibles
    core = core_in.copy()
    factors = factors_in.copy()
    tensor = tensor_in
    
    norm_tensor = tl.norm(tensor, 2)

    # set init if problem
    nb_modes = len(tensor.shape)
    if sparsity_coefficients == None or len(sparsity_coefficients) != nb_modes + 1:
        print("Irrelevant number of sparsity coefficient (different from the number of modes + 1 for the core), they have been set to None.")
        sparsity_coefficients = [None for i in range(nb_modes + 1)]
    if fixed_modes == None:
        fixed_modes = []
    if normalize == None or len(normalize) != nb_modes + 1:
        print("Irrelevant number of normalization booleans (different from the number of modes + 1 for the core), they have been set to False.")
        normalize = [False for i in range(nb_modes + 1)]

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
        if deterministic:
            core, factors, rec_error = one_ntd_step(unfolded_tensors, ranks, core, factors, norm_tensor,
                                          sparsity_coefficients, fixed_modes, normalize, hals = hals, alpha = math.inf)
        else:
            core, factors, rec_error = one_ntd_step(unfolded_tensors, ranks, core, factors, norm_tensor,
                                          sparsity_coefficients, fixed_modes, normalize, hals = hals)
        
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
        return core, np.array(factors), rec_errors, toc
    else:
        return core, np.array(factors)


def one_ntd_step(unfolded_tensors, ranks, in_core, in_factors, norm_tensor,
                 sparsity_coefficients, fixed_modes, normalize, hals = False,
                 alpha=0.5, delta=0.01):
    """
    One pass of Hierarchical Alternating Least Squares update along all modes,
    and hals or gradient update on the core (depends on hals parameter),
    which decreases reconstruction error in Nonnegative Tucker Decomposition.

    Update the factors by solving a least squares problem per mode, as described in [1].

    Note that the unfolding order is the one described in [2], which is different from [1].

    This function is strictly superior to a least squares solver ran on the
    matricized problems min_X ||Y - AX||_F^2 since A is structured as a
    Kronecker product of the other factors/core.
    
    Tensors are manipulated with the tensorly toolbox [3].

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
    sparsity_coefficients: list of float (as much as the number of modes + 1 for the core)
        The sparsity coefficients on each factor and on the core respectively.
    fixed_modes: list of integers (between 0 and the number of modes + 1 for the core)
        Has to be set not to update a factor, taken in the order of modes and lastly on the core.
    normalize: list of boolean (as much as the number of modes + 1 for the core)
        A boolean whereas the factors need to be normalized.
        The normalization is a l_2 normalization on each of the rank components
        (For the factors, each column will be normalized, ie each atom of the dimension of the current rank).
    hals: boolean
        Whether to run hals (true) or gradient (false) update on the core.
        Default (and recommanded): false
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
    core: tensorly tensor
        The core tensor linking the factors of the decomposition
    factors: list of factors
        An array containing all the factors computed with the NTD
    error: float
        The reconstruction error after this update.

    References
    ----------
    [1] Tamara G Kolda and Brett W Bader. "Tensor decompositions and applications",
    SIAM review 51.3 (2009), pp. 455{500.

    [2] Jeremy E Cohen. "About notations in multiway array processing",
    arXiv preprint arXiv:1511.01306, (2015).
    
    [3] J. Kossai et al. "TensorLy: Tensor Learning in Python",
    arxiv preprint (2018)
    """

    # Avoiding errors
    for fixed_value in fixed_modes:
        sparsity_coefficients[fixed_value] = None

    # Copy
    core = in_core.copy()
    factors = in_factors.copy()

    # Generating the mode update sequence
    gen = [mode for mode in range(len(unfolded_tensors)) if mode not in fixed_modes]

    for mode in gen:

        unfolded_core = tl.base.unfold(core, mode)

        tic = time.time()

        # Computing the Kronekcer product
        kron = tl.tenalg.kronecker(factors, skip_matrix = mode, reverse = False)
        kron_core = tl.dot(kron, tl.transpose(unfolded_core))
        rhs = tl.dot(unfolded_tensors[mode], kron_core)
        
        # Maybe suboptimal
        cross = tl.dot(tl.transpose(kron_core), kron_core)

        timer = time.time() - tic

        # Call the hals resolution with nnls, optimizing the current mode
        factors[mode] = tl.transpose(nnls.hals_nnls_acc(tl.transpose(rhs), cross, tl.transpose(factors[mode]),
               maxiter=100, atime=timer, alpha=alpha, delta=delta,
               sparsity_coefficient = sparsity_coefficients[mode], normalize = normalize[mode])[0])

    tensor_shape = tuple([i.shape[0] for i in factors])
    core_shape = tuple(ranks)
    
    refolded_tensor = tl.base.fold(unfolded_tensors[0], 0, tensor_shape)
    
    # Core update
    all_MtX = tl.tenalg.multi_mode_dot(refolded_tensor, factors, transpose = True)
    all_MtM = np.array([fac.T@fac for fac in factors])
    
    if hals:
        # HALS
        AtZ = tl.base.tensor_to_vec(all_MtX)
        AtZ = AtZ.reshape(AtZ.shape[0], 1)
        AtA = tl.tenalg.kronecker(all_MtM)
        vectorized_core = tl.base.tensor_to_vec(core)
        # TODO: nnls.hals.acc is not written for vector inputs? Maybe make a specific function.
        vectorized_core = nnls.hals_nnls_acc(AtZ, AtA, vectorized_core.reshape(vectorized_core.shape[0], 1),
                    maxiter=100, atime=timer, alpha=alpha, delta=delta,
                    sparsity_coefficient = sparsity_coefficients[-1], normalize = False)[0]
        core = vectorized_core.reshape(core_shape)

    else:
        # Projected gradient
        gradient_step = 1
        
        for MtM in all_MtM:
            gradient_step *= 1/(scipy.sparse.linalg.svds(MtM, k = 1)[1][0])
            
        gradient_step = round(gradient_step, 6) # Heurisitc, to avoid consecutive imprecision
        
        cnt = 1
        upd_0 = 0
        upd = 1
        
        if sparsity_coefficients[-1] is None:
            sparse = 0
        else:
            sparse = sparsity_coefficients[-1]
            
        # TODO: dynamic stopping criterion
        # Maybe: try fast gradient instead of gradient            
        while cnt <= 300 and upd>= delta * upd_0:
            gradient = - all_MtX + tl.tenalg.multi_mode_dot(core, all_MtM, transpose = False) + sparse * tl.ones(core.shape)

            # Proposition of reformulation for error computations
            delta_core = np.minimum(gradient_step*gradient, core)
            core = core - delta_core
            upd = tl.norm(delta_core)
            if cnt == 1:
                upd_0 = upd
                
            cnt += 1

    if normalize[-1]: # Only on last mode for now, to be parametrized
        for idx_mat in range(len(core[0,0,:])):
            if np.linalg.norm(core[:,:,idx_mat], 2) != 0:
                core[:,:,idx_mat] = core[:,:,idx_mat] / np.linalg.norm(core[:,:,idx_mat], 2)
    
    # Adding the l1 norm value to the reconstruction error
    sparsity_error = 0
    for index, sparse in enumerate(sparsity_coefficients):
        if sparse:
            if index < len(factors):
                sparsity_error += 2 * (sparse * np.linalg.norm(factors[index], ord=1))
            elif index == len(factors):
                sparsity_error += 2 * (sparse * tl.norm(core, 1))
            else:
                raise NotImplementedError("TODEBUG: Too many sparsity coefficients, should have been raised before.")

    # rec_error = norm_tensor ** 2 - 2*tl.tenalg.inner(all_MtX, core) + tl.tenalg.inner(tl.tenalg.multi_mode_dot(core, all_MtM, transpose = False), core)
    # rec_error = rec_error ** (1/2) + sparsity_error / norm_tensor

    exhaustive_rec_error = (tl.norm(refolded_tensor - tl.tenalg.multi_mode_dot(core, factors, transpose = False), 2) + sparsity_error) / norm_tensor
    # print("diff: " + str(rec_error - exhaustive_rec_error))
    # print("max" + str(np.amax(factors[2])))
    return core, factors, exhaustive_rec_error
