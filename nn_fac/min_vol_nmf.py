##### MINIMUM-VOLUME NMF
## Will be integrated into nmf.py when stable.
import numpy as np
import time
import warnings

import nn_fac.update_rules.min_vol_mu as min_vol_mu
import nn_fac.update_rules.mu as mu
import nn_fac.utils.beta_divergence as beta_div
from nn_fac.utils.normalize_wh import normalize_WH
import nn_fac.utils.errors as err
import nn_fac.utils.initialize_factors as init_factors

eps = 1e-12 # np.finfo(np.float64).eps

#min_vol_computation:
# - "gamma": compute the update with respect to article: V. Leplat, N. Gillis and A.M.S. Ang, "Blind Audio Source Separation with Minimum-Volume Beta-Divergence NMF", IEEE Transactions on Signal Processing 68, pp. 3400-3410, 2020.
# - "lagrangian": compute the update with respect to article: V. Leplat, N. Gillis and J. Idier, "Multiplicative Updates for NMF with β-Divergences under Disjoint Equality Constraints", SIAM Journal on Matrix Analysis and Applications 42.2 (2021), pp. 730-752.

def minvol_beta_nmf(data, rank, beta, n_iter_max = 100, tol = 1e-8, delta = 0.01, lambda_init = 1, min_vol_computation = "gamma", gamma = 1, tol_update_lagrangian = 1e-6, init = "random", W_0 = None, H_0 = None, verbose = False, deterministic = False, seed = 0):
    assert beta in [0,1,2], "This function is only implemented for beta = 2 (Euclidean distance), 1 (Kullback-Leibler divergence), and 0 (Itakura-Saïto divergence)."
    assert min_vol_computation == "gamma", "Only gamma line search is implemented for now, the Lagragian update is unstable."
    
    # Initialization for W and H
    if init.lower() == "custom":
        if W_0 is None or H_0 is None:
            raise err.CustomNotValidFactors("Custom initialization, but (at least) one factor is set to 'None'")
        W = W_0
        H = H_0
        
    else:
        W, H = init_factors.nmf_initialization(data, rank, init, deterministic=deterministic, seed=seed)

    return compute_minvol_beta_nmf(data=data, W_0=W, H_0=H, rank=rank, beta=beta,n_iter_max=n_iter_max, tol=tol, delta=delta, lambda_init=lambda_init, min_vol_computation=min_vol_computation, gamma=gamma, tol_update_lagrangian=tol_update_lagrangian, verbose = verbose)

def compute_minvol_beta_nmf(data, W_0, H_0, rank, beta, n_iter_max = 100, tol = 1e-8, delta = 0.01, lambda_init = 1, min_vol_computation = "gamma", gamma = 1, tol_update_lagrangian = 1e-6, verbose = False):
    assert beta in [0,1,2], "This function is only implemented for beta = 2 (Euclidean distance), 1 (Kullback-Leibler divergence), and 0 (Itakura-Saïto divergence)."
    assert min_vol_computation == "gamma", "Only gamma line search is implemented for now, the Lagragian update is unstable."

    # Algorithm Beta-NMF with logdet(W^t @ W + delta*I) regularization
    W, H = W_0.copy(), H_0.copy()

    # Initialization for loop parameters
    logdetSave = []
    # traceSave = [] # Not useful for now, ask Valentin why them may be
    # condNumberSave = [] # Not useful for now, ask Valentin why them may be

    # Array to save the value of the loss function
    cost_fct_vals = []
    toc = []

    # Initialization for lambda
    log_det = compute_log_det(W, delta)
    lambda_ = lambda_init * beta_div.beta_divergence(data, W @ H, beta) / (log_det + eps)

    # Optimization loop
    for iteration in range(n_iter_max):
        tic = time.time()

        # if min_vol_computation == "gamma": # Keeping only gamma for now, because the Lagrangian is unstable
        if iteration < 6: # No gamma line search for the first 5 iterations
            W, H, err, log_det, _ = one_step_minvol_beta_nmf_gamma(data=data, W=W, H=H, rank=rank, beta=beta, delta=delta, lambda_=lambda_, gamma=None, prev_error=None) # , trace, cond_number
        else:
            W, H, err, log_det, gamma = one_step_minvol_beta_nmf_gamma(data=data, W=W, H=H, rank=rank, beta=beta, delta=delta, lambda_=lambda_, gamma=gamma, prev_error=cost_fct_vals[-1]) # , trace, cond_number
        # elif min_vol_computation == "lagrangian": # Lagrangian is unstable for now. Waiting on future developments from Valentin
        #     W, H, err, log_det, trace, cond_number = one_step_minvol_beta_nmf_lagrangian(data=data, W=W, H=H, rank=rank, beta=beta, delta=delta, alpha=alpha, lambda_=lambda_, tol_update_lagrangian=tol_update_lagrangian)

        toc.append(time.time() - tic)

        cost_fct_vals.append(err)
        logdetSave.append(log_det)
        # condNumberSave.append(cond_number) # Not useful for now, ask Valentin why them may be
        # traceSave.append(trace) # Not useful for now, ask Valentin why them may be

        if verbose:
            if iteration == 0:
                print(f'Normalized cost function value={err}')
            else:
                if cost_fct_vals[-2] - cost_fct_vals[-1] > 0:
                    print(f'Normalized cost function value={cost_fct_vals[-1]}, variation={cost_fct_vals[-2] - cost_fct_vals[-1]}.')
                else:
                    # print in red when the reconstruction error is negative (shouldn't happen)
                    print('\033[91m' + 'Normalized cost function value={}, variation={}.'.format(
                            cost_fct_vals[-1], cost_fct_vals[-2] - cost_fct_vals[-1]) + '\033[0m')

        if iteration > 0 and abs(cost_fct_vals[-2] - cost_fct_vals[-1]) < tol:
            # Stop condition: relative error between last two iterations < tol
            if verbose:
                print(f'Converged in {iteration} iterations.')
            break

    return W, H, cost_fct_vals, toc

def one_step_minvol_beta_nmf_gamma(data, W, H, rank, beta, delta, lambda_, gamma, prev_error):
    assert beta in [0,1,2], "This function is only implemented for beta = 2 (Euclidean distance), 1 (Kullback-Leibler divergence), and 0 (Itakura-Saïto divergence)."

    # Initialize W for gamma line search
    W_prev = W.copy() if gamma is not None else None

    H = mu.switch_alternate_mu(data, W, H, beta, "H") # np.transpose(multiplicative_updates.mu_betadivmin(H.T, W.T, data.T, beta))

    # Update W
    if beta == 0:
        W_update, Y = min_vol_mu.IS_mu_min_vol(data=data, W=W, H=H, delta=delta, lambda_=lambda_)
    elif beta == 1:
        W_update, Y = min_vol_mu.KL_mu_min_vol(data=data, W=W, H=H, delta=delta, lambda_=lambda_)
    elif beta == 2:
        W_update, Y = min_vol_mu.euc_mu_min_vol(data=data, W=W, H=H, delta=delta, lambda_=lambda_)

    # Simplex projection wasn't working, so we turn to standard normlization instead (TODO: check avec Valentin si c'est bon)
    W_normalized, H_normalized = normalize_WH(W_update, H, "W")

    if gamma is not None:
        W, H, gamma = gamma_line_search(data, W_update = W_update, W_gamma_init = W_normalized, W_prev=W_prev, H_gamma_init = H_normalized, 
                                                   beta=beta, delta=delta, gamma_init=gamma, lambda_tilde=lambda_, prev_error=prev_error)
    else:
        W = W_normalized
        H = H_normalized
                
    # Compute the loss function
    log_det = compute_log_det(W, delta)
    err = beta_div.beta_divergence(data, W @ H, beta) + lambda_ * log_det

    # trace = np.trace(Y @ (W.T @ W))
    # cond_number = np.linalg.cond(W.T @ W + delta * np.eye(rank))

    return W, H, err, log_det, gamma #, trace, cond_number


# Lagrangian update, not stable
def one_step_minvol_beta_nmf_lagrangian(data, W, H, rank, beta, delta, alpha, lambda_, tol_update_lagrangian):
    assert beta == 1, "This function is only implemented for beta = 1 (KL divergence)."

    H = mu.switch_alternate_mu(data, W, H, beta, "H") # np.transpose(multiplicative_updates.mu_betadivmin(H.T, W.T, data.T, beta))

    W, Y = min_vol_mu.KL_mu_min_vol_lagrangian(data, W, H, delta, lambda_, tol_update_lagrangian)

    # Compute the loss function
    log_det = compute_log_det(W, delta)
    err = beta_div.beta_divergence(data, W @ H, beta) + lambda_ * log_det

    # trace = np.trace(Y @ (W.T @ W)) # To ask Valentin why it is useful
    # cond_number = np.linalg.cond(W.T @ W + delta * np.eye(rank))

    return W, H, err, log_det #, trace, cond_number


# %% Utils
def gamma_line_search(data, W_update, W_gamma_init, H_gamma_init, beta, delta, gamma_init, lambda_tilde, W_prev, prev_error):
    W_gamma = W_gamma_init.copy()
    H_gamma = H_gamma_init.copy()
    gamma = gamma_init
    cur_log_det = compute_log_det(W_gamma, delta)
    cur_err = beta_div.beta_divergence(data, W_gamma @ H_gamma, beta) + lambda_tilde * cur_log_det
    while cur_err > prev_error and gamma > 1e-16:
        gamma *= 0.8
        W_gamma = (1 - gamma) * W_prev + gamma * W_update
        W_gamma, H_gamma = normalize_WH(W_gamma, H_gamma, "W")
        cur_log_det = compute_log_det(W_gamma, delta)
        cur_err = beta_div.beta_divergence(data, W_gamma @ H_gamma, beta) + lambda_tilde * cur_log_det
    
    gamma = min(gamma * 1.2, 1)
    return W_gamma, H_gamma, gamma

def compute_det(W, delta):
    r = W.shape[1]
    det = np.linalg.det(W.T @ W + delta * np.eye(r))
    return det

def compute_log_det(W, delta):
    det = compute_det(W, delta)
    return np.log10(det, where=(det!=0))

# %% Tester
if __name__ == "__main__":
    np.random.seed(42)
    m, n, rank = 100, 200, 5
    W_0, H_0 = np.random.rand(m, rank), np.random.rand(rank, n) # Example input matrices
    data = W_0@H_0 + 1e-2*np.random.rand(m,n) + eps  # Example input matrix
    
    W, H, lossfun, t = minvol_beta_nmf(data, rank, beta = 1, n_iter_max = 100, min_vol_computation = "gamma", verbose = False)
    print(f"Time KL: {np.sum(t)}, initial error: {lossfun[0]}, final loss func: {lossfun[-1]}")

    W, H, lossfun, t = minvol_beta_nmf(data, rank, beta = 0, n_iter_max = 100, min_vol_computation = "gamma", verbose = False)
    print(f"Time IS: {np.sum(t)}, initial error: {lossfun[0]}, final loss func: {lossfun[-1]}")

    W, H, lossfun, t = minvol_beta_nmf(data, rank, beta = 2, n_iter_max = 100, min_vol_computation = "gamma", verbose = False)
    print(f"Time Euclidean: {np.sum(t)}, initial error: {lossfun[0]}, final loss func: {lossfun[-1]}")

    # TO DEBUG
    # W, H, lossfun, t = minvol_beta_nmf(data, rank, beta = 1, n_iter_max = 100, init = "nndsvd", min_vol_computation = "gamma", verbose = True)
    # W, H, lossfun, t = minvol_beta_nmf(data, rank, beta = 1, n_iter_max = 100, init = "nndsvd", min_vol_computation = "lagrangian", verbose = True)
