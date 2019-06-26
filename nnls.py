# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:40:44 2019

@author: amarmore
"""

import numpy as np
import time

# Accelerated version of algorithm resolving approximately the Nonnegative Least Square problem by computing a Hierarchical .
def hals_nnls_acc(UtM, UtU, in_V, maxiter=500, atime=None, alpha=0.5, delta=0.01,
                  sparsity_coefficient = None, normalize = False, nonzero = False):
## Author : Axel Marmoret, based on Jeremy Cohen version's of Nicolas Gillis Matlab's code for HALS

    """
    =================================
    Non Negative Least Squares (NNLS)
    =================================
    
    Computes an approximate solution of a nonnegative least
    squares problem (NNLS) with an exact block-coordinate descent scheme.
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.
      
    The NNLS unconstrained problem, as defined in [1], solve the following problem:

            min_{V >= 0} ||M-UV||_F^2
    
    The matrix V is updated linewise.
    
    The update rule of the k-th line of V (V[k,:]) for this resolution is:
            V[k,:]_(j+1) = V[k,:]_(j) + (UtM[k,:] - UtU[k,:] V_(j))/UtU[k,k]
      
    This problem can also be defined by adding a sparsity coefficient,
    enhancing sparsity in the solution [2]. The problem thus becomes:
      
            min_{V >= 0} ||M-UV||_F^2 + 2*sparsity_coefficient*(\sum\limits_{j = 0}^{r}||V[k,:]||_1)
    
    NB: 2*sp for uniformization in the derivative
    
    In this sparse version, the update rule for V[k,:] becomes:
            V[k,:]_(j+1) = V[k,:]_(j) + (UtM[k,:] - UtU[k,:] V_(j) - sparsity_coefficient)/UtU[k,k]
    
    This algorithm is defined in [1], as an accelerated version of the HALS algorithm.

    It features two accelerations: an early stop stopping criterion, and a
    complexity averaging between precomputations and loops, so as to use large
    precomputations several times.
    
    This function is made for being used repetively inside an
    outer-loop alternating algorithm, for instance for computing nonnegative
    matrix Factorization or tensor factorization.
    
    Parameters
    ----------
    UtM: r-by-n array
        Pre-computed product of the transposed of U and M, used in the update rule
    UtU: r-by-r array
        Pre-computed product of the transposed of U and U, used in the update rule
    in_V: r-by-n initialization matrix (mutable)
        Initialized V array
        By default, is initialized with one non-zero entry per column
        corresponding to the closest column of U of the corresponding column of M.
    maxiter: Postivie integer
        Upper bound on the number of iterations
        Default: 500
    atime: Positive float
        Time taken to do the precomputations UtU and UtM
        Default: None
    alpha: Positive float
        Ratio between outer computations and inner loops, typically set to 0.5 or 1.
        Default: 0..5
    delta : float in [0,1]
        early stop criterion, while err_k > delta*err_0. Set small for
        almost exact nnls solution, or larger (e.g. 1e-2) for inner loops
        of a PARAFAC computation.
    sparsity_coefficient: float or None
        The coefficient controling the sparisty level in the objective function.
        If set to None, the problem is solved unconstrained.
        Default: None
    normalize: boolean
        True in order to normalize each of the k-th line of V after the update
        False not to update them
        Default: False        
    nonzero: boolean
        True if the lines of the V matrix can't be zero,
        False if they can be zero
        Default: False
    
    Returns
    -------
    V: array
        a r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2
    eps: float
        number of loops authorized by the error stop criterion
    cnt: integer
        final number of update iteration performed
    rho: float
        number of loops authorized by the time stop criterion
    
    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.
    
    [2] J. Eggert, and E. Korner. "Sparse coding and NMF."
    2004 IEEE International Joint Conference on Neural Networks
    (IEEE Cat. No. 04CH37541). Vol. 4. IEEE, 2004.

    """

    r, n = np.shape(UtM)
    if not in_V.size:  # checks if V is empty
        V = np.linalg.linalg.solve(UtU, UtM)  # Least squares
    
        V[V < 0] = 0
        # Scaling
        scale = np.sum(UtM * V)/np.sum(
            UtU * np.dot(V, np.transpose(V)))
        V = np.dot(scale, V)
    else:
        V = in_V.copy()

    rho = 100000
    eps0 = 0
    cnt = 1
    eps = 1

    # Start timer
    tic = time.time()
    while eps >= delta * eps0 and cnt <= 1+alpha*rho and cnt <= maxiter:
        nodelta = 0
        for k in range(r):
            
            if UtU[k,k] != 0:

                if sparsity_coefficient != None: # Using the sparsifying objective function
                    deltaV = np.maximum((UtM[k,:] - UtU[k,:]@V - sparsity_coefficient * np.ones(n)) / UtU[k,k], -V[k,:])
                    V[k,:] = V[k,:] + deltaV

                else:
                    deltaV = np.maximum((UtM[k,:]- UtU[k,:]@V) / UtU[k,k],-V[k,:]) # Element wise maximum -> good idea ?        
                    V[k,:] = V[k,:] + deltaV
                                            
                nodelta = nodelta + np.dot(deltaV, np.transpose(deltaV))
                
                # Safety procedure, if columns aren't allow to be zero
                if nonzero and (V[k,:] == 0).all() :
                    V[k,:] = 1e-16*np.max(V)
                    
            elif nonzero:
                raise ValueError("Column " + str(k) + " of U is zero with nonzero condition")
            
            if normalize:
                norm = np.linalg.norm(V[k,:])
                if norm != 0:
                    V[k,:] /= norm
                else:
                    sqrt_n = 1/n ** (1/2)
                    V[k,:] = [sqrt_n for i in range(n)]
        
        if cnt == 1:
            eps0 = nodelta
            # End timer for one iteration
            btime = max(time.time() - tic, 10e-7) # Avoid division by 0
            
            if atime:  # atime is provided
                # Number of loops authorized
                rho = atime/btime
        eps = nodelta
        cnt += 1
        
    return V, eps, cnt, rho





# NNLS resolution while fitting another matrix
def hals_coupling_nnls_acc(UtM, UtU, in_V, Vtarget, mu,
                           maxiter=500, atime=None, alpha=0.5, delta=0.01,
                           normalize = False, nonzero = False):
    
    """
    ==========================================================
    Non Negative Least Squares (NNLS) with coupling constraint
    ==========================================================
    
    Computes an approximate solution of a nonnegative least
    squares problem (NNLS) with an exact block-coordinate descent scheme.
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.
    The used NNLS resolution algorithm problem is defined in [1],
    and is an accelerated HALS algorithm.
                
    It features two accelerations: an early stop stopping criterion, and a
    complexity averaging between precomputations and loops, so as to use large
    precomputations several times.
    
    This function is made for being used repetively inside an
    outer-loop alternating algorithm, for instance for computing nonnegative
    matrix Factorization or tensor factorization.
    
    Nonetheless, this version is adapted for coupling the returned matrix
    to a second matrix, called Vtarget.
    The optimization problem is defined for PARAFAC2 in [2] as below:

            min_{V >= 0} ||M-UV||_F^2 + mu * ||V - Vtarget||_F^2
    
    Parameters
    ----------
    UtM: r-by-n array
        Pre-computed product of the transposed of U and M, used in the update rule
    UtU: r-by-r array
        Pre-computed product of the transposed of U and U, used in the update rule
    in_V: r-by-n initialization matrix (mutable)
        Initialized V array
        By default, is initialized with one non-zero entry per column
        corresponding to the closest column of U of the corresponding column of M.
    Vtarget: array
        The matrix for V to approach
    mu: float
        The weight given to coupling in the objective function
    maxiter: Postivie integer
        Upper bound on the number of iterations
        Default: 500
    atime: Positive float
        Time taken to do the precomputations UtU and UtM
        Default: None
    alpha: Positive float
        Ratio between outer computations and inner loops, typically set to 0.5 or 1.
        Default: 0..5
    delta : float in [0,1]
        early stop criterion, while err_k > delta*err_0. Set small for
        almost exact nnls solution, or larger (e.g. 1e-2) for inner loops
        of a PARAFAC computation.
    normalize: boolean
        True in order to normalize each of the k-th line of V after the update
        False not to update them
        Default: False        
    nonzero: boolean
        True if the lines of the V matrix can't be zero,
        False if they can be zero
        Default: False
    
    Returns
    -------
    V: array
        a r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2 + mu * ||V - Vtarget||_F^2
    eps: float
        number of loops authorized by the error stop criterion
    cnt: integer
        final number of update iteration performed
    rho: float
        number of loops authorized by the time stop criterion
    
    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.
    
    [2] J. E. Cohen and R. Bro, Nonnegative PARAFAC2: A Flexible Coupling Approach,
    DOI: 10.1007/978-3-319-93764-9_9

    """

    r, n = np.shape(UtM)
    if not in_V.size:  # checks if V is empty
        V = np.linalg.linalg.solve(UtU, UtM)  # Least squares
        V[V < 0] = 0
        # Scaling
        scale = np.sum(UtM * V)/np.sum(
            UtU * np.dot(V, np.transpose(V)))
        V = np.dot(scale, V)
    else:
        V = in_V.copy()

    rho = 100000
    eps0 = 0
    cnt = 1
    eps = 1

    # Start timer
    tic = time.time()
    while cnt <= maxiter and eps >= delta * eps0 and cnt <= 1+alpha*rho:
        nodelta = 0
        for k in range(r):
            
            if UtU[k,k] != 0:
                # Update
                deltaV = np.maximum((UtM[k,:]-UtU[k,:]@V + mu*(Vtarget[k,:] - V[k,:])) / (UtU[k,k] + mu),-V[k,:])
                
                V[k,:] = V[k,:] + deltaV
                
                # Direct update of V
                #V[k,:] = np.maximum((UtM[k,:]-UtU[k,:]@V + UtU[k,k]*V[k,:] + mu*Vtarget[k,:]) / (UtU[k,k] + mu), 0)
                
                nodelta = nodelta + np.dot(deltaV, np.transpose(deltaV))
                
                if nonzero and (V[k,:] == 0).all() :
                    # Safety procedure if we don't want a column to be zero
                    V[k,:] = 1e-16*np.max(V)
            
            elif nonzero:
                raise ValueError("Column " + str(k) + " is zero with nonzero condition")
                
            if normalize:
                norm = np.linalg.norm(V[k,:])
                if norm != 0:
                    V[k,:] /= norm
                else:
                    sqrt_n = 1/n ** (1/2)
                    V[k,:] = [sqrt_n for i in range(n)]
                
        if cnt == 1:
            eps0 = nodelta
            # End timer for one iteration
            btime = max(time.time() - tic, 10e-7) # Avoid division by 0
            
            if atime:  # atime is provided
                # Number of loops authorized
                rho = atime/btime
        eps = nodelta
        cnt = cnt+1

    return V, eps, cnt, rho









# Accelerated version, using gross-tier stopping criterion
def BETA_hals_sparse_smooth_nnls_acc(UtM, UtU, in_V, LtL_in, sp = 1e-7, sm = 1e-7,
                                     maxiter=500, atime=None, alpha=0.5, delta=0.01,
                                     normalize = False, nonzero = False):
    ## Author : Axel Marmoret, based on Jeremy Cohen version of Nicolas Gillis Matlab' code for HALS

    """
    ========================================
    Non Negative Least Squares (NNLS)
    with sparsity and smoothness constraints
    ========================================
    
    Computes an approximate solution of a nonnegative least
    squares problem (NNLS) with an exact block-coordinate descent scheme.
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.
    The used NNLS resolution algorithm problem is defined in [1],
    and is an accelerated HALS algorithm.
                
    It features two accelerations: an early stop stopping criterion, and a
    complexity averaging between precomputations and loops, so as to use large
    precomputations several times.
    
    This function is made for being used repetively inside an
    outer-loop alternating algorithm, for instance for computing nonnegative
    matrix Factorization or tensor factorization.
    
    This algorithm add two constraints over the shape of V: sparsity and smoothness.
    The optimization problem is defined as below (see [2] for instance):

            min_{V >= 0} ||M-UV||_F^2 + 2*sp * (\sum\limits_{j = 0}^{r}||V[k,:]||_1) + sm* (\sum\limits_{j = 0}^{r} ||L V[k,:]||_2^2)
            
    with L the smoothness matrix.
    
    Parameters
    ----------
    UtM: r-by-n array
        Pre-computed product of the transposed of U and M, used in the update rule
    UtU: r-by-r array
        Pre-computed product of the transposed of U and U, used in the update rule
    in_V: r-by-n initialization matrix (mutable)
        Initialized V array
        By default, is initialized with one non-zero entry per column
        corresponding to the closest column of U of the corresponding column of M.
    LtL: array
        The matrix for coupling the factors enhancing smoothness (tridiagonal in general)
    sp: float
        The weight given to sparsity in the objective function
    sm: float
        The weight given to smoothness in the objective function
    maxiter: Postivie integer
        Upper bound on the number of iterations
        Default: 500
    atime: Positive float
        Time taken to do the precomputations UtU and UtM
        Default: None
    alpha: Positive float
        Ratio between outer computations and inner loops, typically set to 0.5 or 1.
        Default: 0..5
    delta : float in [0,1]
        early stop criterion, while err_k > delta*err_0. Set small for
        almost exact nnls solution, or larger (e.g. 1e-2) for inner loops
        of a PARAFAC computation.
    normalize: boolean
        True in order to normalize each of the k-th line of V after the update
        False not to update them
        Default: False        
    nonzero: boolean
        True if the lines of the V matrix can't be zero,
        False if they can be zero
        Default: False
    
    Returns
    -------
    V: array
        a r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2 + mu * ||V - Vtarget||_F^2
    eps: float
        number of loops authorized by the error stop criterion
    cnt: integer
        final number of update iteration performed
    rho: float
        number of loops authorized by the time stop criterion
    
    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.
    
    [2] Kimura, T., & Takahashi, N. (2017). Gauss-Seidel HALS Algorithm for 
    Nonnegative Matrix Factorization with Sparseness and Smoothness Constraints.
    IEICE Transactions on Fundamentals of Electronics, 
    Communications and Computer Sciences, 100(12), 2925-2935.

    """

    r, n = np.shape(UtM)
    if not in_V.size:  # checks if V is empty
        V = np.linalg.linalg.solve(UtU, UtM)  # Least squares
        V[V < 0] = 0
        # Scaling
        scale = np.sum(UtM * V)/np.sum(
            UtU * np.dot(V, np.transpose(V)))
        V = np.dot(scale, V)
    else:
        V = in_V.copy()
        
    LtL = LtL_in.copy()
    
    invLtLUtU = []
    
    for k in range(r):
        invLtLUtU.append(np.invert(UtU[k,k] * np.identity(LtL.shape[0]) + sm * LtL))
    
    rho = 100000
    eps0 = 0
    cnt = 1
    eps = 1

    # Start timer
    tic = time.time()
    while eps >= delta * eps0 and cnt <= 1+alpha*rho and cnt <= maxiter:
        nodelta = 0
        for k in range(r):
            # Update
            if UtU[k,k] != 0:

                """left_member = UtU[k,k] * np.identity(LtL_in.shape[0]) + sm * LtL
                right_member = UtM[k,:] - UtU[k,:]@V + UtU[k,k]@V[k,:] - sp * np.ones(UtM.shape[1]) - sm * np.transpose(LtL@np.transpose(V[k,:]))
                V[k,:] = np.maximum(np.linalg.solve(left_member, right_member),0)"""
                
                modif = np.maximum(((UtM[k,:] - UtU[k,:]@V + UtU[k,k]@V[k,:] - sp * np.ones(UtM.shape[1]))@invLtLUtU[k]), 0)
                
                deltaV = modif - V[k,:]

                V[k,:] = modif

                nodelta = nodelta + np.dot(deltaV, np.transpose(deltaV))
                
                if nonzero and (V[k,:] == 0).all() :
                    # Safety procedure if we don't want a column to be zero
                    V[k,:] = 1e-16*np.max(V)
            elif nonzero:
                raise ValueError("Column " + str(k) + " is zero with nonzero condition")
            
            if normalize:
                norm = np.linalg.norm(V[k,:])
                if norm != 0:
                    V[k,:] /= norm
                else:
                    sqrt_n = 1/n ** (1/2)
                    V[k,:] = [sqrt_n for i in range(n)]

        if cnt == 1:
            eps0 = nodelta
            # End timer for one iteration
            btime = max(time.time() - tic, 10e-7) # Avoid division by 0
            
            if atime:  # atime is provided
                # Number of loops authorized
                rho = 1 + atime/btime #(btime + atime)/btime
        eps = nodelta
        cnt += 1

    return V, eps, cnt, rho


def create_L(rank):
    """
    L matrix for the calculus of the l2 norm of a column of H in the smoothness criteria
    (see Kimura, T., & Takahashi, N. (2017). Gauss-Seidel HALS Algorithm for Nonnegative Matrix Factorization with Sparseness and Smoothness Constraints. IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences, 100(12), 2925-2935.)
    """
    L = np.zeros((rank - 2, rank))
    for i in range(rank - 2):
        L[i,i] = -1
        L[i,i+1] = 2
        L[i,i+2] = -1
    return L


# Different techniques for sparsity enhancement in NNLS
def BETA_hals_sparse_nnls_acc(UtM, UtU, in_V, sparsity, sparsity_coefficient, maxiter=500, atime=None, alpha=0.5, delta=0.01,
                              normalize = False, nonzero = False):
    ## Author : Axel Marmoret, based on Jeremy Cohen version of Nicolas Gillis Matlab' code for HALS

    """
    Computes an approximate solution of a nonnegative least
    squares problem (NNLS) with an exact block-coordinate descent scheme. M is m by n, U is m by r,
    and a sparsity coefficient.
     
    If sparsity is set to "penalty", this algorithm solves:

                min_{V >= 0} ||M-UV||_F^2 + 2*sparsity_coefficient*(\sum\limits_{j = 0}^{r}||V[k,:]||_1)
               
               NB: 2*sp for uniformization in the derivative
    
    else, if sparsity is set to "hard", this algorithm solves:
               
               min_{V >= 0} ||M-UV||_F^2
               
               and keeps only the {sparsity_coefficient} highest factors.
               
               !! For that reason, {sparsity_coefficient} needs to be a nonzero integer here.
               
               !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
               If {sparsity_coefficient} is positive, it computes the sparsity on the lines of the V matrix,
               
               else, it computes the sparsity on the rows of the V matrix
               !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    This accelerated function, defined in [1], is made for being used repetively inside an
    outer-loop alternating algorithm, for instance for computing nonnegative
    matrix Factorization or tensor factorization.

    It features two accelerations: an early stop stopping criterion, and a
    complexity averaging between precomputations and loops, so as to use large
    precomputations several times.

    Parameters
    ----------
    UtM: r-by-n array
        Pre-computed product of the transposed of U and M, used in the update rule
    UtU: r-by-r array
        Pre-computed product of the transposed of U and U, used in the update rule
    in_V: r-by-n initialization matrix (mutable)
        Initialized V array
        By default, is initialized with one non-zero entry per column
        corresponding to the closest column of U of the corresponding column of M.
    sparsity : string
       the sparsity method:
           "penalty" for a sparsity with a penalty coefficient in the objective function,
           "hard" for a hard sparsity where only the highest coefficient are kept
    sparsity_coefficient : float
       the sparsity coefficient, related to the kind of sparsity:
           if "penalty", it will be the coefficient in the objective function (see above)
           if "hard", it will be the number of coefficient to keep
               on columns if > 0
               on rows if < 0
    maxiter: Postivie integer
        Upper bound on the number of iterations
        Default: 500
    atime: Positive float
        Time taken to do the precomputations UtU and UtM
        Default: None
    alpha: Positive float
        Ratio between outer computations and inner loops, typically set to 0.5 or 1.
        Default: 0..5
    delta : float in [0,1]
        early stop criterion, while err_k > delta*err_0. Set small for
        almost exact nnls solution, or larger (e.g. 1e-2) for inner loops
        of a PARAFAC computation.
    normalize: boolean
        True in order to normalize each of the k-th line of V after the update
        False not to update them
        Default: False        
    nonzero: boolean
        True if the lines of the V matrix can't be zero,
        False if they can be zero
        Default: False

    Returns
    -------
    V: array
        a r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2 + mu * ||V - Vtarget||_F^2
    eps: float
        number of loops authorized by the error stop criterion
    cnt: integer
        final number of update iteration performed
    rho: float
        number of loops authorized by the time stop criterion

    References
    ----------
    [1] N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.
    
    [2] J. Eggert, and E. Korner. "Sparse coding and NMF."
    2004 IEEE International Joint Conference on Neural Networks
    (IEEE Cat. No. 04CH37541). Vol. 4. IEEE, 2004.
    """

    r, n = np.shape(UtM)
    if not in_V.size:  # checks if V is empty
        V = np.linalg.linalg.solve(UtU, UtM)  # Least squares
    
        V[V < 0] = 0
        # Scaling
        scale = np.sum(UtM * V)/np.sum(
            UtU * np.dot(V, np.transpose(V)))
        V = np.dot(scale, V)
    else:
        V = in_V.copy()

    rho = 100000
    eps0 = 0
    cnt = 1
    eps = 1
    
    nb_time_bins = 1

    if V[0,:].shape[0] != 1:
        nb_time_bins = V[0,:].shape[0]
    else:
        nb_time_bins = V[0,:].shape[1]
        
    if sparsity == "hard" and not (type(sparsity_coefficient) == int): # Sparsifying the columns with the hard method
        raise ValueError("In the case of hard sparsity, the sparsity_coefficient needs to be an integer.")

    # Start timer
    tic = time.time()
    while eps >= delta * eps0 and cnt <= 1+alpha*rho and cnt <= maxiter:
        nodelta = 0
        for k in range(r):
            
            if UtU[k,k] != 0:
            # Update by penalty term
                if sparsity == "penalty": # Modifying the objective function for sparsification
                    
                    modif = np.true_divide(np.subtract(np.subtract(UtM[k,:], np.dot(UtU[k,:], V)), sparsity_coefficient), UtU[k,k])
                    deltaV = np.maximum(modif,-V[k,:]) # Element wise maximum -> good idea ?
                    
                    V[k,:] = V[k,:] + deltaV

                elif sparsity == "hard": # Sparsifying the rows with the hard method
                    
                    # Classic update
                    deltaV = np.maximum((UtM[k,:]-np.dot(UtU[k,:], V)) / UtU[k,k],-V[k,:]) # Element wise maximum -> good idea ?
        
                    V[k,:] = V[k,:] + deltaV
                    
                    if sparsity_coefficient and sparsity_coefficient > 0:
                        V[k,:] = np.where(V[k,:] < np.percentile(V[k,:],(100-(sparsity_coefficient*100/nb_time_bins))), 0, V[k,:]) # Forcing saprsity by keeping only the {sparsity_coefficient} highest values
                        
                elif sparsity == "power":
                    # Classic update
                    deltaV = np.maximum((UtM[k,:]-np.dot(UtU[k,:], V)) / UtU[k,k],-V[k,:]) # Element wise maximum -> good idea ?
        
                    V[k,:] = V[k,:] + deltaV
                    
                    if sparsity_coefficient and sparsity_coefficient > 0:
                        V[k,:] = keep_most_powerful(V[k,:], sparsity_coefficient) #○ Keep only 95% of the power of the spectrogram
                    
                else:
                    raise ValueError(str(sparsity) + " is not a valid sparsity argument")
                    
                nodelta = nodelta + np.dot(deltaV, np.transpose(deltaV))
                
                # Safety procedure, if columns aren't allow to be zero
                if nonzero and (V[k,:] == 0).all() :
                    V[k,:] = 1e-16*np.max(V)
                
                if normalize:
                    norm = np.linalg.norm(V[k,:])
                    if norm != 0:
                        V[k,:] /= norm
                    else:
                        sqrt_n = 1/n ** (1/2)
                        V[k,:] = [sqrt_n for i in range(n)]
                    
            elif nonzero:
                raise ValueError("Column " + str(k) + " is zero with nonzero condition")
        
        # If we want to sparsify the columns and not the rows
        if sparsity == "hard" and (type(sparsity_coefficient) == int) and sparsity_coefficient < 0 and (-sparsity_coefficient) < r:
            for i in range(nb_time_bins):
                V[:,i] = np.where(V[:,i] < np.percentile(V[:,i],(100+(sparsity_coefficient*100/r))), 0, V[:,i])
        
        elif sparsity == "power" and (type(sparsity_coefficient) == int) and sparsity_coefficient < 0:
            for i in range(nb_time_bins):
                V[:,i] = keep_most_powerful(V[:,i], -sparsity_coefficient) #○ Keep only {sparsity_coefficient}% of the power of the spectrogram

        if cnt == 1:
            eps0 = nodelta
            # End timer for one iteration
            btime = max(time.time() - tic, 10e-7) # Avoid division by 0
            
            if atime:  # atime is provided
                # Number of loops authorized
                rho = atime/btime
        eps = nodelta
        cnt += 1
        
    return V, eps, cnt, rho

def keep_most_powerful(data, percentage):
    somme = 0
    the_test = np.array(data.copy())
    norm_of_ref = np.linalg.norm(the_test, ord=1)
    current_max = 0
    while(somme < percentage * norm_of_ref / 100):
        current_max = np.amax(the_test)
        somme += current_max
        the_test = np.delete(the_test,np.argmax(the_test))
    return np.where(data < current_max, 0, data)
