# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:47:00 2026

@author: bhilaire
"""
#TODO : update functions descriptions

import numpy as np
import time
import math
import warnings
import matplotlib.pyplot as plt

import nn_fac.update_rules.nnls as nnls
import nn_fac.update_rules.mu as mu
import nn_fac.utils.beta_divergence as beta_div
import nn_fac.utils.errors as err
import nn_fac.utils.initialize_factors as init_factors

def hnmf(data, rank, E, init = "random", S_0 = None, A_0 = None, H_0 = None, n_iter_max=100, n_stepS=100, n_stepH=100, tol=1e-8,
        update_rule = "hals", beta = 2,
        sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False, False],
        verbose=False, return_costs=False, deterministic=False, rev_order=False, seed=0):
    """
    ======================================
    Nonnegative Matrix Factorization (NMF)
    ======================================

    Factorization of a matrix M in two nonnegative matrices U and V,
    such that the product UV approximates M.
    If M is of size m*n, U and V are respectively of size m*r and r*n,
    r being the rank of the decomposition (parameter)
    Typically, this method is used as a dimensionality reduction technique,
    or for source separation.

    The objective function is:

        d(M - UV)_{\beta}
        + sparsity_coefficients[0] * (\sum\limits_{j = 0}^{r}||U[:,k]||_1)
        + sparsity_coefficients[1] * (\sum\limits_{j = 0}^{r}||V[k,:]||_1)

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
        Threshold on the improvement in cost function value.
        Between two succesive iterations, if the difference between 
        both cost function values is below this threshold, the algorithm stops.
        Default: 1e-8
    update_rule: string "hals" | "mu"
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
    sparsity_coefficients: List of float (two)
        The sparsity coefficients on U and V respectively.
        If set to None, the algorithm is computed without sparsity
        Default: [None, None],
    fixed_modes: List of integers (between 0 and 2)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: []
    normalize: List of boolean (two)
        Indicates whether the factors need to be normalized or not.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise for U, linewise for V)
        Default: [False, False]
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

    Returns
    -------
    U, V: numpy arrays
        Factors of the NMF
    cost_fct_vals: list
        A list of the normalized cost function values, for every iteration of the algorithm.
    toc: list
        A list with accumulated time for every iterations

    Example
    -------
    >>> import numpy as np
    >>> from nn_fac import nmf
    >>> rank = 5
    >>> U_lines = 100
    >>> V_col = 125
    >>> U_0 = np.random.rand(U_lines, rank)
    >>> V_0 = np.random.rand(rank, V_col)
    >>> M = U_0@V_0
    >>> U, V = nmf.nmf(M, rank, init = "random", n_iter_max = 500, tol = 1e-8,
               sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False],
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
        np.random.seed(seed)

    if init.lower() == "custom":
        if S_0 is None or H_0 is None:
            raise err.CustomNotValidFactors("Custom initialization, but (at least) one factor is set to 'None'")
        
    else:
        #S_0, H_0 = init_factors.nmf_initialization(data, rank, init, deterministic=deterministic, seed=seed) 
        #only makes sense with random for now.
        f = data.shape(0)
        t = data.shape(1)
        S_0 = np.random.rand(f,rank) + 1e-12
        H_0 = np.random.rand(rank,t) + 1e-12
        A_0 = np.ones((rank,rank))

    return compute_hnmf(data, rank, S_0, E, A_0, H_0, n_iter_max=n_iter_max, n_stepS=n_stepS, n_stepH=n_stepH, tol=tol,
                       update_rule = update_rule, beta = beta,
                       sparsity_coefficients = sparsity_coefficients, fixed_modes = fixed_modes, normalize = normalize,
                       verbose=verbose, return_costs=return_costs, deterministic=deterministic, rev_order=rev_order)

def compute_hnmf(data, rank, S_in, E_in, A_in, H_in, n_iter_max=100, n_stepS=100, n_stepH=100, tol=1e-8,
                update_rule = "hals", beta = 2,
                sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False, False],
                verbose=False, return_costs=False, deterministic=False, rev_order=False):
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
    U_in: array of floats
        Initial U factor, of size m*r
    V_in: array of floats
        Initial V factor, of size r*n
    n_iter_max: integer
        The maximal number of iteration before stopping the algorithm
        Default: 100
    tol: float
        Threshold on the improvement in cost function value.
        Between two iterations, if the difference between 
        both cost function values is below this threshold, the algorithm stops.
        Default: 1e-8
    update_rule: string "hals" | "mu"
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
    sparsity_coefficients: List of float (two)
        The sparsity coefficients on U and V respectively.
        If set to None, the algorithm is computed without sparsity
        Default: [None, None],
    fixed_modes: List of integers (between 0 and 2)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: []
    normalize: List of boolean (two)
        Indicates whether the factors need to be normalized or not.
        The normalization is a l_2 normalization on each of the rank components
        (columnwise for U, linewise for V)
        Default: [False, False]
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

    Returns
    -------
    U, V: numpy arrays
        Factors of the NMF
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
    H = H_in.copy()
    E = E_in.copy()
    A = A_in.copy()
    cost_fct_vals = []
    norm_data = np.linalg.norm(data)
    tic = time.time()
    toc = []

    if sparsity_coefficients == None:
        sparsity_coefficients = [None, None]
    if fixed_modes == None:
        fixed_modes = []
    if normalize == None or normalize == False:
        normalize = [False, False, False]

    for iteration in range(n_iter_max):

        # One pass of least squares on each updated mode
        S, H, A, cost = one_hnmf_step(data, rank, S, E, A, H, norm_data, update_rule, beta,
                                sparsity_coefficients, fixed_modes, normalize, deterministic, rev_order=rev_order, n_stepS=n_stepS, n_stepH=n_stepH)

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
        return np.array(S), np.array(H), np.array(A), cost_fct_vals, toc
    else:
        return np.array(S), np.array(H), np.array(A)

def one_hnmf_step(data, rank, S_in, E, A_in, H_in, norm_data, update_rule, beta,
                 sparsity_coefficients, fixed_modes, normalize, deterministic, rev_order=False, n_stepS=100, n_stepH=100):
    """
    Adding the possibility of choosing the number of nnls steps and the order of updates
    """
    if update_rule not in ["hals", "mu"]:
        raise err.InvalidArgumentValue(f"Invalid update rule: {update_rule}") from None
    if update_rule == "hals" and beta != 2:
        raise err.InvalidArgumentValue(f"The hals is only valid for the frobenius norm, corresponding to the beta divergence with beta = 2. Here, beta was set to {beta}. To compute NMF with this value of beta, please use the mu update_rule.") from None

    if len(sparsity_coefficients) != 2:
        raise ValueError("NMF needs 2 sparsity coefficients to be performed")

    # Copy
    S = S_in.copy()
    A = A_in.copy()
    H = H_in.copy()

    #V = np.random.rand(rank, len(V[0,:])) + 1e-12

    def update_S(X):
        Hp = np.dot(np.multiply(E,A),H)
        if 0 not in fixed_modes:
            
            if update_rule == "hals":
                # Set timer for acceleration in hals_nnls_acc
                tic = time.time()
        
                # End timer for acceleration in hals_nnls_acc
                timer = time.time() - tic
        
                # Compute HALS/NNLS resolution
                if deterministic:
                    Y = np.transpose(nnls.hals_nnls_acc_test(np.transpose(data), np.transpose(Hp), np.transpose(X), maxiter=n_stepS, atime=timer, alpha=math.inf, delta=0.01,
                                                    sparsity_coefficient = sparsity_coefficients[0], normalize = normalize[0], nonzero = False, return_costs=False)[0])
                else:
                    Y = np.transpose(nnls.hals_nnls_acc_test(np.transpose(data), np.transpose(Hp), np.transpose(X), maxiter=n_stepS, atime=timer, alpha=0.5, delta=0.01,
                                                    sparsity_coefficient = sparsity_coefficients[0], normalize = normalize[0], nonzero = False, return_costs=False)[0])
            
            elif update_rule == "mu":
                Y = mu.switch_alternate_mu(data, X, Hp, beta, "U") #mu.mu_betadivmin(U, V, data, beta)
                
                if normalize[0]:
                    n, _ = np.shape(data)
                    _, r = np.shape(Y)
                    for k in range(r):
                        norm = np.linalg.norm(Y[:,k])
                        if norm != 0:
                            Y[:,k] /= norm
                        else:
                            sqrt_n = 1/n ** (1/2)
                            Y[:,k] = [sqrt_n for _ in range(n)]
                            #assert False
            """
            Y = mu.switch_alternate_mu(data, X, Hp, beta, "U", eps=0)
            if normalize[0]:
                n, _ = np.shape(data)
                _, r = np.shape(Y)
                for k in range(r):
                    norm = np.linalg.norm(Y[:,k])
                    if norm != 0:
                        Y[:,k] /= norm
                    else:
                        sqrt_n = 1/n ** (1/2)
                        Y[:,k] = [sqrt_n for _ in range(n)]
                        #assert False
            """
            """
            S_id_cqt_Bpo36_allnotes = np.zeros((288,88))
            bin = 0
            for note in range(88):
                S_id_cqt_Bpo36_allnotes[bin,note] = 1.0
                S_id_cqt_Bpo36_allnotes[bin+1,note] = 1.0
                S_id_cqt_Bpo36_allnotes[bin+2,note] = 1.0
                bin += 3
            Y += 1e-12
            Y = np.multiply(Y, S_id_cqt_Bpo36_allnotes)
            """
            return Y #/np.max(Y)
        else :
            return X
    
    def update_H(X):
        W = np.dot(S, np.multiply(E,A))
        if 2 not in fixed_modes:
            if update_rule == "hals":
                # Set timer for acceleration in hals_nnls_acc
                tic = time.time()
        
                # End timer for acceleration in hals_nnls_acc
                timer = time.time() - tic
        
                # Compute HALS/NNLS resolution
                if deterministic:
                    Y,_,_,_ = nnls.hals_nnls_acc_test(data, W, X, maxiter=n_stepH, atime=timer, alpha=math.inf, delta=0.01,
                                    sparsity_coefficient = sparsity_coefficients[1], normalize = normalize[2], nonzero = False, return_costs=False)
                    #plt.plot(costs_H_updt)
                    #plt.show()
                else:
                    Y = nnls.hals_nnls_acc_test(data, W, X, maxiter=n_stepH, atime=timer, alpha=0.5, delta=0.01,
                                    sparsity_coefficient = sparsity_coefficients[1], normalize = normalize[2], nonzero = False, return_costs=False)[0]
            
            elif update_rule == "mu":
                Y = mu.switch_alternate_mu(data, W, X, beta, "V") # np.transpose(mu.mu_betadivmin(V.T, U.T, data.T, beta))

                if normalize[2]:
                    _, n = np.shape(data)
                    r, _ = np.shape(Y)
                    for k in range(r):
                        norm = np.linalg.norm(Y[k,:])
                        if norm != 0:
                            Y[k,:] /= norm
                        else:
                            sqrt_n = 1/n ** (1/2)
                            Y[k,:] = [sqrt_n for _ in range(n)]
            return Y
        else :
            return X
        
    def update_A(X, eta=0.05, l2_p=True):
        if 1 not in fixed_modes:
            #Only a multiplicative update is implemented for now
            StDHt = E * np.dot(np.transpose(S), np.dot(data, np.transpose(H)))
            #WARNING : the numerator and denominator are multiplied pointwise by E, which is sparse. The simplification can only occur for elements where E
            #is nonnegative.
            StS = np.dot(np.transpose(S),S)
            HHt = np.dot(H,np.transpose(H))
            ApE = np.multiply(X,E)
            Denom = E * np.dot(StS, np.dot(ApE, HHt))
            if l2_p: #adding l2 norm penalty
                Denom += 2*X 
            not_fonda_mask = ((np.ones((rank,rank)) - np.identity(rank)) != 0)

            non_zero_mask = ((np.multiply(Denom, ApE)) != 0) #Update where E and A, but also the denominator are not zero.

            ##we remove the diagonal from the A values to update :
            #mask = np.logical_and(not_fonda_mask, not_zero_mask)

            frac = np.ones((rank,rank))
            np.divide(StDHt, Denom, out=frac, where=non_zero_mask) #MUR style ; problem is there might be some zeros in the denom since E is very sparse
            Y = np.maximum(1e-12, np.multiply(X, frac, out=X.copy(), where=not_fonda_mask))
            #Y = np.maximum(0, X - eta*(Denom - StDHt))                    #Gradient descent classic + projection to keep positive constraint up.

            if normalize[1]:
                for k in range(rank):
                    norm = np.linalg.norm(Y[:,k])
                    if norm != 0:
                        Y[:,k] /= norm
                    else:
                        sqrt_n = 1/rank ** (1/2)
                        Y[:,k] = [sqrt_n for _ in range(rank)]
            
            return Y
        else :
            return X
    
    if rev_order:
        H = update_H(H)
        S = update_S(S)
        A = update_A(A)
    
    else :
        S = update_S(S)
        H = update_H(H)
        A = update_A(A)
    
    sparsity_coefficients = np.where(np.array(sparsity_coefficients) == None, 0, sparsity_coefficients)
    
    if update_rule == "hals":
        cost = np.linalg.norm(data-np.dot(np.dot(S,np.multiply(E,A)),H), ord='fro') ** 2 #+ 2 * (sparsity_coefficients[0] * np.linalg.norm(U, ord=1) + sparsity_coefficients[1] * np.linalg.norm(V, ord=1))
        #print(f"S maxvalue : {np.max(S)}")
        #print(f"H maxvalue : {np.max(H)}")

    elif update_rule == "mu":
        cost = beta_div.beta_divergence(data, np.dot(np.dot(S,np.multiply(E,A)),H), beta)

    #cost = cost/(norm_data**2
    return S, H, A, cost

if __name__ == "__main__":
    np.random.seed(42)
    m, n, rank = 100, 200, 5
    W_0, H_0 = np.random.rand(m, rank), np.random.rand(rank, n) # Example input matrices
    data = W_0@H_0 + 1e-2*np.random.rand(m,n)  # Example input matrix
    """
    W, H = nmf(data, rank, beta = 2, update_rule = "hals", n_iter_max = 100, init="random", verbose = True)
    W, H = nmf(data, rank, beta = 2, update_rule = "hals", n_iter_max = 100, init = "nndsvd",verbose = True)

    W, H = nmf(data, rank, beta = 1, update_rule = "mu", n_iter_max = 100, init="random", verbose = True)
    W, H = nmf(data, rank, beta = 1, update_rule = "mu", n_iter_max = 100, init = "nndsvd",verbose = True)

    W, H = nmf(data, rank, beta = 0, update_rule = "mu", n_iter_max = 100, init="random", verbose = True)
    W, H = nmf(data, rank, beta = 0, update_rule = "mu", n_iter_max = 100, init = "nndsvd",verbose = True)
    """
