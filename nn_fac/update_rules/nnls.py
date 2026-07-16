# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:40:44 2019

@author: amarmore
"""

import math
import tensorly as tl
import time

import nn_fac.utils.errors as err

########################################
# High-level functions
########################################

def switch_alternate_hals(data, U, V, matrix, maxiter=100, sparsity_coefficient = None, normalize = False, nonzero = False, hals_inner_tol=1e-8):
    """
    High-level wrapper around hals_nnls that handles both factor updates (U and V)
    in a NMF/NTD context, hiding the transposition bookkeeping.

    For the U update, the problem min_{U>=0} ||data - U V||^2 is recast as
    min_{U.T>=0} ||data.T - V.T U.T||^2 so that hals_nnls always solves for
    the *right* factor, which avoids duplicating logic.

    Parameters
    ----------
    data : m-by-n array
    U    : m-by-r array
    V    : r-by-n array
    matrix : "U" | "W" to update U, or "V" | "H" to update V
    maxiter, sparsity_coefficient, normalize, nonzero, hals_inner_tol :
        Forwarded to hals_nnls.

    Returns
    -------
    Updated factor matrix (same shape as the requested factor).
    """
    match matrix:
        case "U" | "W":
            return tl.transpose(hals_nnls(tl.transpose(data), tl.transpose(V), tl.transpose(U), maxiter=maxiter, sparsity_coefficient = sparsity_coefficient, normalize = normalize, nonzero = nonzero, hals_inner_tol=hals_inner_tol)[0])
        
        case "V" | "H":
            return hals_nnls(data, U, V, maxiter=maxiter, sparsity_coefficient = sparsity_coefficient, normalize = normalize, nonzero = nonzero, hals_inner_tol=hals_inner_tol)[0]
        
        case _:
            raise err.InvalidArgumentValue(f"Invalid value for matrix: got {matrix}, but it must be 'U' or 'W' for the first matrix, and 'V' or 'H' for the second one.") from None

def switch_alternate_hals_acc(data, U, V, matrix, maxiter=100, alpha=0.5, delta=0.01,
                              sparsity_coefficient = None, normalize = None, nonzero = False,
                              deterministic = False):
    """
    Same as switch_alternate_hals but using the time-accelerated solver
    hals_nnls_acc instead of hals_nnls.

    When deterministic=True the time-based acceleration is disabled
    (alpha is set to inf) so that results are reproducible across runs.

    Parameters
    ----------
    data : m-by-n array
    U    : m-by-r array (used as fixed factor or as warm-start depending on `matrix`)
    V    : r-by-n array (used as fixed factor or as warm-start depending on `matrix`)
    matrix : "U" | "W" to update U, or "V" | "H" to update V
    maxiter, alpha, delta, sparsity_coefficient, normalize, nonzero :
        Forwarded to hals_nnls_acc.
    deterministic : bool
        If True, overrides alpha=inf to disable the timing heuristic.

    Returns
    -------
    Updated factor matrix (same shape as the requested factor).
    """
    if deterministic:
        alpha=math.inf # Override alpha to ensure deterministic behavior, as the acceleration is based on computation time, which may vary.
    match matrix:
        case "U" | "W":
            return tl.transpose(hals_nnls_acc(tl.transpose(data), tl.transpose(V), tl.transpose(U), maxiter=maxiter, alpha=alpha, delta=delta, sparsity_coefficient = sparsity_coefficient, normalize = normalize, nonzero = nonzero)[0])
        
        case "V" | "H":
            return hals_nnls_acc(data, U, V, maxiter=maxiter, alpha=alpha, delta=delta, sparsity_coefficient = sparsity_coefficient, normalize = normalize, nonzero = nonzero)[0]
        
        case _:
            raise err.InvalidArgumentValue(f"Invalid value for matrix: got {matrix}, but it must be 'U' or 'W' for the first matrix, and 'V' or 'H' for the second one.") from None


def vector_nnls_solver(y, A, x, maxiter=500, sparsity_coefficient = None, normalize = False, nonzero = False, hals_inner_tol=1e-8):
    """
    Convenience wrapper: solve the NNLS problem min_{x>=0} ||y - A x||^2 for a
    single vector x.  Reshapes inputs to 2-D matrices, calls hals_nnls, and
    returns a flat 1-D result.
    """
    y_mat = y.reshape(y.shape[0], 1)
    X = x.reshape(x.shape[0], 1)
    
    X_up, eps, cnt = hals_nnls(y_mat, A, X, maxiter=maxiter,sparsity_coefficient = sparsity_coefficient, normalize = normalize, nonzero = nonzero, hals_inner_tol=hals_inner_tol)
    
    return X_up.reshape(X_up.shape[0])

########################################
# HALS without accelerations
########################################

def hals_nnls(in_data, in_U, in_V, maxiter=100, sparsity_coefficient = None, normalize = False, nonzero = False, hals_inner_tol=1e-8):
    """
    Solve the NNLS problem  min_{V >= 0} ||in_data - in_U V||_F^2  via HALS.

    Iterates one_step_hals_nnls until either maxiter is reached or the
    cumulative squared update falls below hals_inner_tol.

    Parameters
    ----------
    in_data : m-by-n array  — the data matrix M
    in_U    : m-by-r array  — the fixed factor U
    in_V    : r-by-n array  — initialisation for V
    maxiter : int           — maximum number of passes
    sparsity_coefficient : float or None
        L1 penalty coefficient; None means no sparsity.
    normalize : bool  — normalise each row of V after each update
    nonzero   : bool  — raise if any row of V becomes zero
    hals_inner_tol : float
        Stop early when the cumulative squared update is below this threshold.

    Returns
    -------
    V            : r-by-n nonnegative matrix
    idx_iter     : last iteration index
    updates_cumulative_sum : update magnitude at the last iteration
    """
    # Validate input shapes and dimensions
    if len(in_data.shape) != 2:
        raise err.ArgumentException(f"Argument data is an array of {len(in_data.shape)} dimensions when it should be a matrix.")
    if len(in_U.shape) != 2:
        raise err.ArgumentException(f"Argument U is an array of {len(in_U.shape)} dimensions when it should be a matrix.")
    if len(in_V.shape) != 2:
        raise err.ArgumentException(f"Argument V is an array of {len(in_V.shape)} dimensions when it should be a matrix.")
    if in_data.shape[0] != in_U.shape[0]:
        raise err.ArgumentException("data and U must have the same number of rows")
    if in_data.shape[1] != in_V.shape[1]:
        raise err.ArgumentException("data and V must have the same number of columns")
    if in_U.shape[1] != in_V.shape[0]:
        raise err.ArgumentException("U and V must have the same number of columns and rows respectively")

    data = in_data.copy()
    U = in_U.copy()
    V = in_V.copy()

    # Precompute the cross products for the inner HALS loop.
    UtU = tl.dot(tl.transpose(U),U)
    UtM = tl.dot(tl.transpose(U),data)

    return compute_hals_nnls(V, UtM, UtU, maxiter=maxiter, sparsity_coefficient = sparsity_coefficient, normalize = normalize, nonzero = nonzero, hals_inner_tol=hals_inner_tol)

def compute_hals_nnls(V, UtM, UtU, maxiter=100, sparsity_coefficient = None, normalize = False, nonzero = False, hals_inner_tol=1e-8):
    """
    Solve the NNLS problem  min_{V >= 0} ||in_data - in_U V||_F^2  via HALS.

    Iterates one_step_hals_nnls until either maxiter is reached or the
    cumulative squared update falls below hals_inner_tol.

    Parameters
    ----------
    V   : r-by-n matrix (updated in-place and returned)
    UtM : r-by-n pre-computed product U^T M
    UtU : r-by-r pre-computed product U^T U
    maxiter : int           — maximum number of passes
    sparsity_coefficient : float or None
        L1 penalty coefficient; None means no sparsity.
    normalize : bool  — normalise each row of V after each update
    nonzero   : bool  — raise if any row of V becomes zero
    hals_inner_tol : float
        Stop early when the cumulative squared update is below this threshold.

    Returns
    -------
    V            : r-by-n nonnegative matrix
    idx_iter     : last iteration index
    updates_cumulative_sum : update magnitude at the last iteration
    """
    # The problem is convex but the nonnegativity constraint means no closed-form
    # solution exists. Iterate until convergence or maxiter is reached.
    for idx_iter in range(maxiter):
        V, updates_cumulative_sum = one_step_hals_nnls(V, UtM, UtU, sparsity_coefficient = sparsity_coefficient, normalize = normalize, nonzero = nonzero)
        if updates_cumulative_sum < hals_inner_tol: # Stop criterion: update is small enough
            break
    return V, idx_iter, updates_cumulative_sum


########################################
# Iterated functions
########################################

def one_step_hals_nnls(V, UtM, UtU, sparsity_coefficient = None, normalize = False, nonzero = False):
    """
    Perform one full pass of the HALS NNLS update over all rows of V.

    Each row k of V is updated by calling columnwise_nnls, which applies the
    block-coordinate descent step for that row while holding all others fixed.

    Parameters
    ----------
    V   : r-by-n matrix (updated in-place and returned)
    UtM : r-by-n pre-computed product U^T M
    UtU : r-by-r pre-computed product U^T U
    sparsity_coefficient, normalize, nonzero : forwarded to columnwise_nnls

    Returns
    -------
    V                     : updated r-by-n matrix
    updates_cumulative_sum : sum of ||V[k,:] - old_V[k,:]||^2 over all k
                            (used as a convergence criterion)
    """
    updates_cumulative_sum = 0
    # Loop over the rows of V
    for k in range(V.shape[0]):
        if UtU[k, k] == 0:
            if nonzero:
                raise err.ZeroColumnWhenUnautorized(f"A column {k} of U (in the nnls update) is zero with nonzero condition")
            continue
        old_Vk = V[k,:].copy()
        V[k,:] = columnwise_nnls(V, UtM, UtU, k, sparsity_coefficient=sparsity_coefficient, normalize=normalize, nonzero=nonzero)
        deltaV = V[k,:] - old_Vk # Compute the change in V[k,:]
        updates_cumulative_sum += tl.dot(deltaV, tl.transpose(deltaV))

    return V, updates_cumulative_sum


def columnwise_nnls(V, UtM, UtU, k, sparsity_coefficient=None, normalize=False, nonzero=False):
    """
    Apply one HALS block-coordinate descent update to row k of V.

    Computes the clipped gradient step for row k, then optionally applies
    a nonzero safety floor and row normalisation.

    Parameters
    ----------
    V : r-by-n matrix
        Row k will be updated; other rows are read but not modified.
    UtM : r-by-n array  — U^T M
    UtU : r-by-r array  — U^T U
    k : int
        Row index to update.
    sparsity_coefficient : float or None
        L1 penalty coefficient. None means no sparsity.
    normalize : bool
        Normalise the updated row to unit norm. Default: False.
    nonzero : bool
        Replace a zero row with a small uniform vector. Default: False.

    Returns
    -------
    updated_col : (n,) nonnegative array  — updated row k of V
    """
    n = V.shape[1]

    # NNLS resolution, with or without sparsity constraint
    if sparsity_coefficient != None: # Using the sparsifying objective function
        deltaV = tl.clip((UtM[k,:] - UtU[k,:]@V - sparsity_coefficient * tl.ones(n)) / UtU[k,k], a_min=-V[k,:])

    else: # Without sparsity constraint
        deltaV = tl.clip((UtM[k,:] - UtU[k,:]@V) / UtU[k,k], a_min=-V[k,:])

    # Make the update
    updated_col = V[k,:] + deltaV

    if nonzero and (updated_col == 0).all(): # Safety procedure, if columns aren't allow to be zero
        updated_col = 1e-10 * tl.ones(n)

    if normalize:
        norm = tl.norm(updated_col)
        if norm != 0:
            updated_col /= norm
        else:
            sqrt_n = 1 / n ** (1 / 2)
            updated_col = sqrt_n * tl.ones(n)

    assert len(updated_col.shape) == 1, f"Updated column has shape {updated_col.shape}, expected a 1D array."
    assert updated_col.shape[0] == n, f"Updated column has shape {updated_col.shape}, expected ({n},)"

    return updated_col
    
########################################
# HALS with accelerations. DEPRECATED
########################################


def hals_nnls_acc(in_data, in_U, in_V, maxiter=500, alpha=0.5, delta=0.01,
                  sparsity_coefficient = None, normalize = False, nonzero = False):
## Author : Axel Marmoret, based on Jeremy Cohen version's of Nicolas Gillis Matlab's code for HALS

    """
    Solve the NNLS problem  min_{V >= 0} ||in_data - in_U V||_F^2  via accelerated HALS.

    Uses a time-based heuristic to balance the cost of the precomputation
    (U^T U, U^T M, measured as atime) against the cost of one HALS pass
    (btime), allowing at most alpha * (atime/btime) additional passes after
    the first.  An early-stop criterion halts iteration when the cumulative
    update magnitude drops below delta times the first-pass magnitude.

    Parameters
    ----------
    in_data : m-by-n array
        The data matrix M.
    in_U : m-by-r array
        The fixed factor U.
    in_V : r-by-n array
        Initialisation for V.
    maxiter : int
        Upper bound on the number of iterations. Default: 500.
    alpha : float
        Ratio between precomputation cost and inner-loop cost; controls how
        many extra passes are allowed. Default: 0.5.
    delta : float in [0, 1]
        Early-stop threshold: stop when update magnitude < delta * eps0.
        Use a small value for a precise solution, larger (e.g. 1e-2) for
        inner loops inside a larger algorithm. Default: 0.01.
    sparsity_coefficient : float or None
        L1 penalty coefficient. None means no sparsity. Default: None.
    normalize : bool
        Normalise each row of V after each update. Default: False.
    nonzero : bool
        Raise if any row of V becomes zero. Default: False.

    Returns
    -------
    V : r-by-n nonnegative matrix
        Approximate solution to argmin_{V >= 0} ||M - UV||_F^2.
    eps : float
        Cumulative update magnitude at the last iteration.
    cnt : int
        Total number of HALS passes performed.

    References
    ----------
    [1] N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [2] J. Eggert and E. Korner, "Sparse coding and NMF",
    IEEE IJCNN, Vol. 4, 2004.
    """
    if len(in_data.shape) != 2:
        raise err.ArgumentException(f"Argument data is an array of {len(in_data.shape)} dimensions when it should be a matrix.")
    if len(in_U.shape) != 2:
        raise err.ArgumentException(f"Argument U is an array of {len(in_U.shape)} dimensions when it should be a matrix.")
    if len(in_V.shape) != 2:
        raise err.ArgumentException(f"Argument V is an array of {len(in_V.shape)} dimensions when it should be a matrix.")
    if in_data.shape[0] != in_U.shape[0]:
        raise err.ArgumentException("data and U must have the same number of rows")
    if in_data.shape[1] != in_V.shape[1]:
        raise err.ArgumentException("data and V must have the same number of columns")
    if in_U.shape[1] != in_V.shape[0]:
        raise err.ArgumentException("U and V must have the same number of columns and rows respectively")

    data = in_data.copy()
    U = in_U.copy()
    V = in_V.copy()

    # Set timer for acceleration in hals_nnls_acc
    tic = time.time()

    # Computing cross products
    UtU = tl.dot(tl.transpose(U),U)
    UtM = tl.dot(tl.transpose(U),data)

    # End timer for acceleration in hals_nnls_acc
    atime = time.time() - tic

    return compute_hals_nnls_acc(V, UtM, UtU, maxiter=maxiter, alpha=alpha, delta=delta,
                            sparsity_coefficient = sparsity_coefficient, normalize = normalize, nonzero = nonzero, atime=atime)

def compute_hals_nnls_acc(V, UtM, UtU, maxiter=500, alpha=0.5, delta=0.01,
                            sparsity_coefficient = None, normalize = False, nonzero = False, atime=NotImplemented):
                          
    # first loop:
    # Start timer
    tic = time.time()
    V, eps0 = one_step_hals_nnls(V, UtM, UtU, sparsity_coefficient=sparsity_coefficient, normalize=normalize, nonzero=nonzero)
    btime = max(time.time() - tic, 10e-7) # Avoid division by 0

    # Compute the maximum number of inner iterations allowed by the time budget.
    # rho ≈ atime / btime: if one HALS pass takes btime seconds and the
    # precomputation took atime seconds, allow ~alpha*rho extra passes.
    rho = 100000  # Default: effectively disables the time-based stopping criterion
    if atime:
        rho = atime / btime

    # Initialise the convergence criterion to the first step's update magnitude,
    # then continue while improvement is still above delta * eps0.
    eps = eps0
    cnt = 1
    while eps >= delta * eps0 and cnt <= 1 + alpha * rho and cnt <= maxiter:
        V, eps = one_step_hals_nnls(V, UtM, UtU, sparsity_coefficient=sparsity_coefficient, normalize=normalize, nonzero=nonzero)
        cnt += 1

    return V, eps, cnt


########################################
# SANDBOX
########################################


#### Sandbox of NNLS, for specials cases (as PARAFAC2 as other constraints than sparsity).
#### This code is flagged as "sandbow" because it's not properly tested.
def hals_coupling_nnls_acc(in_data, in_U, in_V, Vtarget, mu,
                           maxiter=500, alpha=0.5, delta=0.01,
                           normalize = False, nonzero = False):

    """
    Solve the coupled NNLS problem
        min_{V >= 0} ||M - UV||_F^2 + mu * ||V - Vtarget||_F^2
    via accelerated HALS.

    Uses the same time-based acceleration as hals_nnls_acc. The coupling term
    pushes V toward Vtarget with weight mu, which modifies both the numerator
    (extra mu*(Vtarget[k,:] - V[k,:])) and the denominator (UtU[k,k] + mu) of
    the per-row update step.

    Parameters
    ----------
    in_data : m-by-n array
        The data matrix M.
    in_U : m-by-r array
        The fixed factor U.
    in_V : r-by-n array
        Initialisation for V.
    Vtarget : r-by-n array
        Target matrix for the coupling term.
    mu : float
        Coupling weight.
    maxiter : int
        Upper bound on the number of iterations. Default: 500.
    alpha : float
        Time-budget ratio. Default: 0.5.
    delta : float in [0, 1]
        Early-stop threshold. Default: 0.01.
    normalize : bool
        Normalise each row of V after each update. Default: False.
    nonzero : bool
        Raise if any row of V becomes zero. Default: False.

    Returns
    -------
    V : r-by-n nonnegative matrix
        Approximate solution to the coupled problem.
    eps : float
        Cumulative update magnitude at the last iteration.
    cnt : int
        Total number of HALS passes performed.
    rho : float
        Number of iterations authorised by the time criterion.

    References
    ----------
    [1] N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [2] J. E. Cohen and R. Bro, Nonnegative PARAFAC2: A Flexible Coupling Approach,
    DOI: 10.1007/978-3-319-93764-9_9
    """

    if len(in_data.shape) != 2:
        raise err.ArgumentException(f"Argument data is an array of {len(in_data.shape)} dimensions when it should be a matrix.")
    if len(in_U.shape) != 2:
        raise err.ArgumentException(f"Argument U is an array of {len(in_U.shape)} dimensions when it should be a matrix.")
    if len(in_V.shape) != 2:
        raise err.ArgumentException(f"Argument V is an array of {len(in_V.shape)} dimensions when it should be a matrix.")
    if in_data.shape[0] != in_U.shape[0]:
        raise err.ArgumentException("data and U must have the same number of rows")
    if in_data.shape[1] != in_V.shape[1]:
        raise err.ArgumentException("data and V must have the same number of columns")
    if in_U.shape[1] != in_V.shape[0]:
        raise err.ArgumentException("U and V must have the same number of columns and rows respectively")

    data = in_data.copy()
    U = in_U.copy()
    V = in_V.copy()

    # Set timer for acceleration
    tic = time.time()
    UtU = tl.dot(tl.transpose(U), U)
    UtM = tl.dot(tl.transpose(U), data)
    atime = time.time() - tic

    if in_V.size == 0:  # checks if V is empty
        V = tl.solve(UtU, UtM)  # Least squares
        V[V < 0] = 0
        # Scaling
        scale = tl.sum(UtM * V)/tl.sum(
            UtU * tl.dot(V, tl.transpose(V)))
        V = tl.dot(scale, V)

    rho = 100000
    eps0 = 0
    cnt = 1
    eps = 1

    # Start timer
    tic = time.time()
    while cnt <= maxiter and eps >= delta * eps0 and cnt <= 1+alpha*rho:
        V, nodelta = one_step_hals_coupling_nnls(V, UtM, UtU, Vtarget, mu, normalize=normalize, nonzero=nonzero)

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

# NNLS resolution while approaching another matrix
def one_step_hals_coupling_nnls(V, UtM, UtU, Vtarget, mu, normalize=False, nonzero=False):
    """
    One full HALS pass for the coupled NNLS problem
        min_{V >= 0} ||M - UV||_F^2 + mu * ||V - Vtarget||_F^2.

    The coupling term shifts the gradient by mu*(Vtarget[k,:] - V[k,:]) and
    changes the effective step-size denominator to (UtU[k,k] + mu).

    Parameters
    ----------
    V       : r-by-n matrix (updated in-place and returned)
    UtM     : r-by-n  U^T M
    UtU     : r-by-r  U^T U
    Vtarget : r-by-n  target matrix for the coupling term
    mu      : float   coupling weight
    normalize, nonzero : same semantics as in one_step_hals_nnls

    Returns
    -------
    V      : updated r-by-n matrix
    nodelta : sum of ||deltaV[k,:]||^2 over k (convergence criterion)
    """
    nodelta = 0
    n = V.shape[1]
    for k in range(V.shape[0]):
        if UtU[k, k] == 0:
            if nonzero:
                raise err.ZeroColumnWhenUnautorized(f"A column {k} of U (in the coupling nnls update) is zero with nonzero condition")
            continue
        deltaV = tl.clip((UtM[k,:] - UtU[k,:]@V + mu*(Vtarget[k,:] - V[k,:])) / (UtU[k,k] + mu), a_min=-V[k,:])
        V[k,:] = V[k,:] + deltaV
        if nonzero and (V[k,:] == 0).all():
            V[k,:] = 1e-16 * tl.max(V)
        if normalize:
            norm = tl.norm(V[k,:])
            if norm != 0:
                V[k,:] /= norm
            else:
                sqrt_n = 1 / n ** (1/2)
                V[k,:] = sqrt_n * tl.ones(n)
        nodelta += tl.dot(deltaV, tl.transpose(deltaV))
    return V, nodelta