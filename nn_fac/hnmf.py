# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:49:25 2019

@author: amarmore

# Note: not tested with torch backend actually!!!! Only numpy. TODO

"""

import random
import tensorly as tl
import time
import warnings

import nn_fac.update_rules.nnls as nnls
import nn_fac.update_rules.mu as mu
import nn_fac.utils.beta_divergence as beta_div
import nn_fac.utils.errors as err
import nn_fac.utils.initialize_factors as init_factors
import nn_fac.utils.tensorly_additional_utils as tl_additional_utils

def hnmf(data, rank, E, init = "random", S_0 = None, A_0 = None, V_0 = None, n_iter_max=100, n_stepS=100, n_stepV=100, tol=1e-8,
        update_rule = "hals", beta = 2,
        sparsity_coefficients = [None, None, None], fixed_modes = [False, False, False], normalize = [False, False, False],
        update_order = "SV",
        verbose=False, return_costs=False, deterministic=False, seed=0):
    """
    ======================================
    Harmonic Nonnegative Matrix Factorization (HNMF)
    ======================================

    Factorization of a matrix M in four nonnegative matrices S, A, E and V,
    such that the product S(E°A)V approximates M.
    If M is of size m*n, S, E, A and V are respectively of size m*r, r*r, r*r and r*n,
    r being the rank of the decomposition (parameter)
    Typically, this method is used as a dimensionality reduction technique,
    or for source separation with musical applications.

    The objective function is:

        d(M - S(E°A)V)_{\beta}
        + sparsity_coefficients[0] * (\sum\limits_{j = 0}^{r}||S[:,k]||_1)
        + sparsity_coefficients[1] * (\sum\limits_{j = 0}^{r}||A[:,k]||_1)
        + sparsity_coefficients[2] * (\sum\limits_{j = 0}^{r}||V[k,:]||_1)

    With:

        d(A)_{\beta} the elementwise $\beta$-divergence,
        ||a||_1 = \sum_{i} abs(a_{i}) (Elementwise L1 norm)

    The objective function is minimized by fixing alternatively
    one of both factors U and V and optimizing on the other one.
    More precisely, the chosen optimization algorithm is the HALS [1],
    which updates each factor columnwise, fixing every other columns,
    each subproblem being reduced to a Nonnegative Least Squares problem 
    if the update_rule is "hals",
    or by using the Multiplicative Update [4,5] on each factor
    if the update_rule is "mu".
    The MU is minimizes the $\beta$-divergence,
    whereas the HALS minimizes the Frobenius norm only.

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
            This NNDSVD is implemented in the nn_fac.utils.initialize_factors module, based on the nimfa implementation [3].
        - If set to custom:
            U_0 and V_0 (see below) will be used for the initialization
        Default: random
    S_0: None or array of nonnegative floats
        A custom initialization of S, used only in "custom" init mode.
        Default: None
    A_0: None or array of nonnegative floats
        A custom initialization of A, used only in "custom" init mode.
        Default: None
    V_0: None or array of nonnegative floats
        A custom initialization of V, used only in "custom" init mode.
        Default: None
    n_iter_max: integer
        The maximal number of iteration before stopping the algorithm
        Default: 100
    n_stepS: integer
        The maximal number of iteration in the HALS update for the S matrix if the update rule is HALS.
        Default: 100
    n_stepV: integer
        The maximal number of iteration in the HALS update for the V matrix if the update rule is HALS.
        Default: 100
    tol: float
        Threshold on the improvement in cost function value.
        Between two succesive iterations, if the difference between 
        both cost function values is below this threshold, the algorithm stops.
        Default: 1e-8
    update_rule: string "hals" | "hals_acc" | "mu"
        The chosen update rule.
        HALS ("hals") performs optimization with the euclidean norm without
        time-based acceleration.
        Accelerated HALS ("hals_acc") uses the same update rule with a
        timing heuristic to allow more inner iterations when precomputation
        is relatively expensive.
        MU ("mu") performs the optimization using the $\beta$-divergence loss, 
        which generalizes the Euclidean norm, and the Kullback-Leibler and 
        Itakura-Saito divergences.
        The chosen beta-divergence is specified with the parameter `beta`.
        Default: "hals"
    beta: float
        The beta parameter for the beta-divergence.
        2 - Euclidean norm
        1 - Kullback-Leibler divergence
        0 - Itakura-Saito divergence
        Default: 2
    sparsity_coefficients: List of float (three)
        The sparsity coefficients on S, A and V respectively.
        If set to None, the algorithm is computed without sparsity
        Default: [None, None, None],
    fixed_modes: List of integers (between 0 and 2 included)
        Has to be set not to update a factor, 0, 1 and 2 for S, A and V respectively
        Default: [False, False, False]
    normalize: List of boolean (three)
        Indicates whether the factors need to be normalized or not.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise for S and A, linewise for V)
        Default: [False, False]
    update_order: string "SV" | "VS" | "SH" | "HS"
        The order in which the factors are updated. A is always updated last as a convention.
        Default: "UV"
    verbose: boolean
        Indicates whether the algorithm prints the successive
        normalized cost function values or not
        Default: False
    return_costs: boolean
        Indicates whether the algorithm should return all normalized cost function 
        values and computation time of each iteration or not
        Default: False
    deterministic: boolean
        Whether or not the NMF should be computed determinstically (True) or not (False).
        In details, the determinisitc condition covers the initialization 
        and the acceleration condition which is based on timing (and hence not deteministic).
        Default: False
    seed: integer
        The seed for the random number generator, used for the initialization
        and the acceleration condition.
        Default: 0

    Returns
    -------
    S, A, V: numpy arrays
        Factors of the HNMF
    cost_fct_vals: list
        A list of the normalized cost function values, for every iteration of the algorithm.
    toc: list
        A list with accumulated time for every iterations

    Example
    -------
    >>> import numpy as np
    >>> from nn_fac import nmf
    >>> rank = 5
    >>> S_lines = 100
    >>> V_col = 125
    >>> S_0 = np.random.rand(U_lines, rank)
    >>> A_0 = np.random.rand(rank, rank)
    >>> E = np.ones((rank, rank))
    >>> V_0 = np.random.rand(rank, V_col)
    >>> M = S_0@(np.multiply(A,E))@V_0
    >>> S, A, V = hnmf.hnmf(M, rank, E, init = "random", n_iter_max = 500, tol = 1e-8,
               sparsity_coefficients = [None, None, None], fixed_modes = [], normalize = [False, False, False],
               verbose=True, return_costs = False)

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
    
    [4] Févotte, C., & Idier, J. (2011). 
    Algorithms for nonnegative matrix factorization with the β-divergence. 
    Neural computation, 23(9), 2421-2456.
    
    [5] Lee, D. D., & Seung, H. S. (1999). 
    Learning the parts of objects by non-negative matrix factorization.
    Nature, 401(6755), 788-791.
    """
    if min(data.shape) < rank:
        min_data = min(data.shape)
        rank = min_data
        warnings.warn(f"The rank is too high for the input matrix. It was set to {min_data} instead.")

    if deterministic:
        random.seed(seed)
        tl_additional_utils.set_random_state(seed)

    if init.lower() == "custom":
        if S_0 is None or A_0 is None or V_0 is None:
            raise err.CustomNotValidFactors("Custom initialization, but (at least) one factor is set to 'None'")
        
    else:
        S_0, A_0, V_0 = init_factors.hnmf_initialization(data, rank, E, init, deterministic=deterministic, seed=seed)

    return compute_hnmf(data, rank, E, S_0, A_0, V_0, n_iter_max=n_iter_max, n_stepS=n_stepS, n_stepV=n_stepV, tol=tol,
                       update_rule = update_rule, beta = beta,
                       sparsity_coefficients = sparsity_coefficients, fixed_modes = fixed_modes, normalize = normalize,
                       update_order = update_order,
                       verbose=verbose, return_costs=return_costs, deterministic=deterministic)

# Author : Jeremy Cohen, modified by Axel Marmoret and Baptiste Hilaire
def compute_hnmf(data, rank, E, S_in, A_in, V_in, n_iter_max=100, n_stepS=100, n_stepV=100, tol=1e-8,
                update_rule = "hals", beta = 2,
                sparsity_coefficients = [None, None, None], fixed_modes = [False, False, False], normalize = [False, False, False],
                update_order = "SV",
                verbose=False, return_costs=False, deterministic=False):
    """
    Computation of a Nonnegative matrix factorization via
    hierarchical alternating least squares (HALS) [1],
    or Multiplicative Update (MU) [2],
    with U_in and V_in as initialization.

    Parameters
    ----------
    data: nonnegative array
        The matrix M, which is factorized, of size m*n
    rank: integer
        The rank of the decomposition
    E: array of ints
        Matrix of harmonic relations
    S_in: array of floats
        Initial S factor, of size m*r
    A_in: array of floats
        Initial A factor, of size r*r   
    V_in: array of floats
        Initial V factor, of size r*n
    n_iter_max: integer
        The maximal number of iteration before stopping the algorithm
        Default: 100
    n_stepS: integer
        The maximal number of iteration in the HALS update for the S matrix if the update rule is HALS.
        Default: 100
    n_stepV: integer
        The maximal number of iteration in the HALS update for the V matrix if the update rule is HALS.
        Default: 100
    tol: float
        Threshold on the improvement in cost function value.
        Between two iterations, if the difference between 
        both cost function values is below this threshold, the algorithm stops.
        Default: 1e-8
    update_rule: string "hals" | "hals_acc" | "mu"
        The chosen update rule.
        HALS performs optimization with the euclidean norm,
        MU performs the optimization using the $\beta$-divergence loss, 
        which generalizes the Euclidean norm, and the Kullback-Leibler and 
        Itakura-Saito divergences.
        The chosen beta-divergence is specified with the parameter `beta`.
        Default: "hals"
    beta: float
        The beta parameter for the beta-divergence.
        2 - Euclidean norm
        1 - Kullback-Leibler divergence
        0 - Itakura-Saito divergence
        Default: 2
    sparsity_coefficients: List of float (three)
        The sparsity coefficients on S, A and V respectively.
        If set to None, the algorithm is computed without sparsity
        Default: [None, None, None],
    fixed_modes: List of integers (between 0 and 2)
        Has to be set not to update a factor, 0, 1 and 2 for S, A and V respectively
        Default: []
    normalize: List of boolean (three)
        Indicates whether the factors need to be normalized or not.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise for S and A, linewise for V)
        Default: [False, False, False]
    update_order: string "SV" | "VS" | "SH" | "HS"
        The order in which the factors are updated.
        Default: "SV"
    verbose: boolean
        Indicates whether the algorithm prints the successive
        normalized cost function values or not
        Default: False
    return_costs: boolean
        Indicates whether the algorithm should return all normalized cost function 
        values and computation time of each iteration or not
        Default: False
    deterministic: boolean
        Whether or not the NMF should be computed determinstically (True) or not (False).
        In details, the determinisitc condition covers the initialization 
        and the acceleration condition which is based on timing (and hence not deteministic).
        Default: False
    seed: integer
        The seed for the random number generator, used for the initialization
        and the acceleration condition.
        Default: 0

    Returns
    -------
    S, A, V: numpy arrays
        Factors of the HNMF
    cost_fct_vals: list
        A list of the normalized cost function values, for every iteration of the algorithm.
    toc: list
        A list with accumulated time at each iterations

    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.
    
    [2] Févotte, C., & Idier, J. (2011). 
    Algorithms for nonnegative matrix factorization with the β-divergence. 
    Neural computation, 23(9), 2421-2456.
    """
    # initialisation
    S = S_in.copy()
    A = A_in.copy()
    V = V_in.copy()
    cost_fct_vals = []
    norm_data = tl.norm(data)
    tic = time.time()
    toc = []

    if sparsity_coefficients == None:
        sparsity_coefficients = [None, None, None]
    if fixed_modes == None or fixed_modes == []:
        fixed_modes = [False, False, False]
    if normalize == None or normalize == False:
        normalize = [False, False, False]

    for iteration in range(n_iter_max):

        # One pass of least squares on each updated mode
        S, A, V, cost = one_hnmf_step(data, rank, E, S, A, V, n_stepS, n_stepV, norm_data, update_rule, beta,
                                  sparsity_coefficients, fixed_modes, normalize, deterministic, update_order)

        toc.append(time.time() - tic)

        cost_fct_vals.append(cost)

        if verbose:
            if iteration == 0:
                print('Normalized cost function value={}'.format(cost))
            else:
                if cost_fct_vals[-2] - cost_fct_vals[-1] > 0:
                    print('Normalized cost function value={}, variation={}.'.format(
                            cost_fct_vals[-1], cost_fct_vals[-2] - cost_fct_vals[-1]))
                else:
                    # print in red when the reconstruction error is negative (shouldn't happen)
                    print('\033[91m' + 'Normalized cost function value={}, variation={}.'.format(
                            cost_fct_vals[-1], cost_fct_vals[-2] - cost_fct_vals[-1]) + '\033[0m')

        if iteration > 0 and abs(cost_fct_vals[-2] - cost_fct_vals[-1]) < tol:
            # Stop condition: relative error between last two iterations < tol
            if verbose:
                print('Converged in {} iterations.'.format(iteration))
            break

    if return_costs:
        return tl.tensor(S), tl.tensor(A), tl.tensor(V), cost_fct_vals, toc
    else:
        return tl.tensor(S), tl.tensor(A), tl.tensor(V)


def one_hnmf_step(data, rank, E, S_in, A_in, V_in, n_stepS, n_stepV, norm_data, update_rule, beta,
                 sparsity_coefficients, fixed_modes, normalize, deterministic, update_order):
    """
    One pass of updates for each factor in NMF
    Update the factors by solving a nonnegative least squares problem per mode
    if the update_rule is "hals",
    or by using the Multiplicative Update on each factor
    if the update_rule is "mu".

    Parameters
    ----------
    data: nonnegative array
        The matrix M, which is factorized, of size m*n
    rank: integer
        The rank of the decomposition
    S_in: array of floats
        Initial S factor, of size m*r
    A_in: array of floats
        Initial A factor, of size r*r
    V_in: array of floats
        Initial V factor, of size r*n
    n_stepS: integer
        The maximal number of iteration in the HALS update for the S matrix if the update rule is HALS.
    n_stepV: integer
        The maximal number of iteration in the HALS update for the V matrix if the update rule is HALS.
    norm_data: float
        The Frobenius norm of the input matrix (data)
    update_rule: string "hals" | "hals_acc" | "mu"
        The chosen update rule.
        HALS performs optimization with the euclidean norm,
        MU performs the optimization using the $\beta$-divergence loss, 
        which generalizes the Euclidean norm, and the Kullback-Leibler and 
        Itakura-Saito divergences.
        The chosen beta-divergence is specified with the parameter `beta`.
    beta: float
        The beta parameter for the beta-divergence.
        2 - Euclidean norm
        1 - Kullback-Leibler divergence
        0 - Itakura-Saito divergence
    sparsity_coefficients: List of float (three)
        The sparsity coefficients on S, A and V respectively.
        If set to None, the algorithm is computed without sparsity
    fixed_modes: List of integers (between 0 and 2)
        Has to be set not to update a factor, 0, 1 and 2 for S, A and V respectively
    normalize: List of boolean (three)
        A boolean whereas the factors need to be normalized.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise for S and A, linewise for V)
    deterministic: boolean
        Whether or not the NMF should be computed determinstically (True) or not (False).
        In details, the determinisitc condition covers the initialization 
        and the acceleration condition which is based on timing (and hence not deteministic).

    Returns
    -------
    S, A, V: numpy arrays
        Factors of the NMF
    cost_fct_val:
        The value of the cost function at this step,
        normalized by the squared norm of the original matrix.
    """
    # Validate the update rule.
    # "hals" and "hals_acc" both minimise the Frobenius norm (beta=2 only);
    # "mu" uses the beta-divergence and supports arbitrary beta.
    if update_rule not in ["hals", "hals_acc", "mu"]:
        raise err.InvalidArgumentValue(f"Invalid update rule: {update_rule}") from None
    if update_rule in ["hals", "hals_acc"] and beta != 2:
        raise err.InvalidArgumentValue(f"The hals is only valid for the frobenius norm, corresponding to the beta divergence with beta = 2. Here, beta was set to {beta}. To compute NMF with this value of beta, please use the mu update_rule.") from None

    if len(sparsity_coefficients) != 2:
        raise ValueError("NMF needs 2 sparsity coefficients to be performed")

    # Copy
    S = S_in.copy()
    A = A_in.copy()
    V = V_in.copy()

    def S_update(data, E, S, A, V, n_iter, update_rule, beta, sparsity_coefficients, normalize, deterministic):
        match update_rule:
            case "hals":
                tmp_S = nnls.switch_alternate_hals(data, S, (np.multiply(E,A)@V), "U", maxiter=n_iter, sparsity_coefficient = sparsity_coefficients[0], normalize = normalize[0], nonzero = False, hals_inner_tol=1e-8)

            case "hals_acc":
                # switch_alternate_hals_acc already returns the updated factor directly
                # (not a tuple), so no indexing is needed.
                # U_in is used as the warm-start initialisation for the NNLS inner solver.
                tmp_S = nnls.switch_alternate_hals_acc(data, S_in, (np.multiply(E,A)@V), "U", maxiter=n_iter, alpha=0.5, delta=0.01,
                                            sparsity_coefficient = sparsity_coefficients[0], normalize = normalize[0], nonzero=False,
                                            deterministic = deterministic)
            
            case "mu":
                tmp_S = mu.switch_alternate_mu(data, S, (np.multiply(E,A)@V), beta, "U") #mu.mu_betadivmin(U, V, data, beta)

            case _:
                raise err.InvalidArgumentValue(f"Invalid update rule: {update_rule}") from None

        return tmp_S

    def V_update(data, E, S, A, V, n_iter, update_rule, beta, sparsity_coefficients, normalize, deterministic):
        match update_rule:
            case "hals":
                # sparsity and normalize index 1 corresponds to V (index 0 is for U)
                tmp_V = nnls.switch_alternate_hals(data, (S@(np.multiply(E,A))), V, "V", maxiter=n_iter, sparsity_coefficient = sparsity_coefficients[2], normalize = normalize[2], nonzero = False, hals_inner_tol=1e-8)

            case "hals_acc":
                # U is the *current* fixed factor (possibly updated earlier this step);
                # using U_in here would silently apply a stale factor.
                tmp_V = nnls.switch_alternate_hals_acc(data, (S@(np.multiply(E,A))), V, "V", maxiter=n_iter, alpha=0.5, delta=0.01,
                                            sparsity_coefficient = sparsity_coefficients[2], normalize = normalize[2], nonzero=False,
                                            deterministic = deterministic)
            
            case "mu":
                tmp_V = mu.switch_alternate_mu(data, (S@(np.multiply(E,A))), V, beta, "V")

            case _:
                raise err.InvalidArgumentValue(f"Invalid update rule: {update_rule}") from None

        return tmp_V

    def A_update(data, E, S, A, V, sparsity_coefficients, normalize, l2_p=True):

        tmp_A = A

        #Only a multiplicative update is implemented for now
        StDVt = E * np.dot(np.transpose(S), np.dot(data, np.transpose(V)))
        #WARNING : the numerator and denominator are multiplied pointwise by E, which is sparse. The simplification can only occur for elements where E
        #is nonnegative.
        StS = np.dot(np.transpose(S),S)
        VVt = np.dot(H,np.transpose(V))
        ApE = np.multiply(tmp_A,E)
        Denom = E * np.dot(StS, np.dot(ApE, VVt))
        if l2_p: #adding l2 norm penalty
            Denom += 2*tmp_A 
        not_fonda_mask = ((np.ones((rank,rank)) - np.identity(rank)) != 0)

        non_zero_mask = ((np.multiply(Denom, ApE)) != 0) #Update where E and A, but also the denominator are not zero.

        ##we remove the diagonal from the A values to update :
        #mask = np.logical_and(not_fonda_mask, not_zero_mask)

        frac = np.ones((rank,rank))
        np.divide(StDVt, Denom, out=frac, where=non_zero_mask) #MUR style ; problem is there might be some zeros in the denom since E is very sparse
        tmp_A = np.maximum(1e-12, np.multiply(tmp_A, frac, out=A.copy(), where=not_fonda_mask))
        #Y = np.maximum(0, X - eta*(Denom - StDHt))                    #Gradient descent classic + projection to keep positive constraint up.

        if normalize[1]:
            for k in range(rank):
                norm = np.linalg.norm(tmp_A[:,k])
                if norm != 0:
                    tmp_A[:,k] /= norm
                else:
                    sqrt_n = 1/rank ** (1/2)
                    tmp_A[:,k] = [sqrt_n for _ in range(rank)]

        return tmp_A

    if fixed_modes[0] and fixed_modes[1] and fixed_modes[2]:
        raise err.InvalidArgumentValue("All factors are fixed, nothing to update.")
    
    elif fixed_modes[0] and fixed_modes[1]:
        # Only update V
        V = V_update(data, E, S, A, V, n_stepV, update_rule, beta, sparsity_coefficients, normalize, deterministic)

    elif fixed_modes[1] and fixed_modes[2]:
        # Only update U
        S = S_update(data, E, S, A, V, n_stepS, update_rule, beta, sparsity_coefficients, normalize, deterministic)
    
    elif fixed_modes[0] and fixed_modes[2]:
        # Only update A
        A = A_update(data, E, S, A, V, sparsity_coefficients, normalize, l2_p=True)
    
    elif fixed_modes[0]:
        V = V_update(data, E, S, A, V, n_stepV, update_rule, beta, sparsity_coefficients, normalize, deterministic)
        A = A_update(data, E, S, A, V, sparsity_coefficients, normalize, l2_p=True)

    elif fixed_modes[2]:
        S = S_update(data, E, S, A, V, n_stepS, update_rule, beta, sparsity_coefficients, normalize, deterministic)
        A = A_update(data, E, S, A, V, sparsity_coefficients, normalize, l2_p=True)

    else:
        match update_order:
            case "SV" | "SH":
                # S update
                S = S_update(data, E, S, A, V, n_stepS, update_rule, beta, sparsity_coefficients, normalize, deterministic)

                # V update
                V = V_update(data, E, S, A, V, n_stepV, update_rule, beta, sparsity_coefficients, normalize, deterministic)

            case "VS" | "HS":
                # V update
                V = V_update(data, E, S, A, V, n_stepV, update_rule, beta, sparsity_coefficients, normalize, deterministic)
            
                # S update
                S = S_update(data, E, S, A, V, n_stepS, update_rule, beta, sparsity_coefficients, normalize, deterministic)

            case _:
                raise err.InvalidArgumentValue(f"Invalid update order: {update_order}. Should be 'SV', 'VS', 'SH', or 'HS'") from None

        # update A
        if not fixed_modes[1]:
            A = A_update(data, E, S, A, V, sparsity_coefficients, normalize, l2_p=True)

    # Replace None sparsity entries with 0 for cost computation
    sparsity_coefficients = tl.where(tl.tensor(sparsity_coefficients) == None, 0, sparsity_coefficients)
    
    if update_rule in ["hals", "hals_acc"]:  # Both HALS variants minimise the Frobenius norm
        cost = tl.norm(data-tl.dot(tl.dot(S,E*A),V), order=2) ** 2 + 2 * (sparsity_coefficients[0] * tl.norm(S, order=1) + sparsity_coefficients[1] * tl.norm(A, order=1) + sparsity_coefficients[2] * tl.norm(V, order=1))
    
    elif update_rule == "mu":
        cost = beta_div.beta_divergence(data, tl.dot(tl.dot(S,E*A),V), beta)

    #cost = cost/(norm_data**2)
    return S, A, V, cost

if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)
    m, n, rank = 100, 200, 5
    S_0, A_0, E_0, H_0 = np.random.rand(m, rank), np.random.rand(rank, rank), np.random.rand(rank, rank), np.random.rand(rank, n) # Example input matrices
    data = S_0@(np.multiply(A_0, E_0))@H_0 + 1e-2*np.random.rand(m,n)  # Example input matrix
    
    S, A, H = hnmf(data, rank, E_0, beta = 2, update_rule = "hals", n_iter_max = 100, init="random", verbose = True)

    S, A, H = hnmf(data, rank, E_0, beta = 1, update_rule = "mu", n_iter_max = 100, init="random", verbose = True)

    S, A, H = hnmf(data, rank, E_0, beta = 0, update_rule = "mu", n_iter_max = 100, init="random", verbose = True)
  
