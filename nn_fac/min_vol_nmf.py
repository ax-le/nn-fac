##### MINIMUM-VOLUME NMF
## Will be integrated into nmf.py when stable.
import numpy as np
import time
from nimfa.methods import seeding

import nn_fac.update_rules.min_vol_mu as min_vol_mu
import nn_fac.update_rules.mu as mu
import nn_fac.utils.beta_divergence as beta_div
from nn_fac.utils.normalize_wh import normalize_WH
import nn_fac.utils.errors as err

eps = 1e-12 # np.finfo(np.float64).eps

# gamma_line_search:
# - "True": compute the update with respect to article: V. Leplat, N. Gillis and A.M.S. Ang, "Blind Audio Source Separation with Minimum-Volume Beta-Divergence NMF", IEEE Transactions on Signal Processing 68, pp. 3400-3410, 2020.
# - "False": compute the update with respect to article: V. Leplat, N. Gillis and J. Idier, "Multiplicative Updates for NMF with β-Divergences under Disjoint Equality Constraints", SIAM Journal on Matrix Analysis and Applications 42.2 (2021), pp. 730-752.

def minvol_beta_nmf(data, rank, beta, n_iter_max = 100, tol = 1e-8, delta = 0.01, lambda_init = 1, gamma_line_search = False, gamma = 1, tol_update_lagrangian = 1e-6, init = "random", W_0 = None, H_0 = None, verbose = False):
    assert beta == 1, "This function is only implemented for beta = 1 (KL divergence)."
    
    # Initialization for W and H
    if init == 'random':
        m,n = data.shape
        np.random.seed(0)
        W = np.random.rand(m, rank)
        H = np.random.rand(rank, n)
    # elif init.lower() == "nndsvd": #Doesn't work, to debug!
    #     W, H = seeding.Nndsvd().initialize(data, rank, {'flag': 0})
    #     W = np.maximum(W, eps)
    #     H = np.maximum(H, eps)
    elif init == 'custom':
        assert W_0 is not None and H_0 is not None, "You must provide W_0 and H_0 if you want to use the custom initialization."
        W = W_0
        H = H_0
    else:
        raise NotImplementedError(f"This initialization method is not implemented: {init}")
    
    return compute_minvol_beta_nmf(data=data, W_0=W, H_0=H, rank=rank, beta=beta,n_iter_max=n_iter_max, tol=tol, delta=delta, lambda_init=lambda_init, gamma_line_search=gamma_line_search, gamma=gamma, tol_update_lagrangian=tol_update_lagrangian, verbose = verbose)

def compute_minvol_beta_nmf(data, W_0, H_0, rank, beta, n_iter_max = 100, tol = 1e-8, delta = 0.01, lambda_init = 1, gamma_line_search = False, gamma = 1, tol_update_lagrangian = 1e-6, verbose = False):
    assert beta == 1, "This function is only implemented for beta = 1 (KL divergence)."
    if gamma_line_search:
        assert gamma is not None, "You must set gamma if you use the gamma line search strategy."

    # Algorithm Beta-NMF with logdet(W^t @ W + delta*I) regularization
    W, H = W_0.copy(), H_0.copy()

    # Initialization for loop parameters
    traceSave = []
    logdetSave = []
    condNumberSave = []

    # Array to save the value of the loss function
    cost_fct_vals = []
    toc = []

    # Initialization for lambda
    log_det = min_vol_mu.compute_log_det(W, delta)
    lambda_ = lambda_init * beta_div.beta_divergence(data, W @ H, beta) / (log_det + eps)

    # Optimization loop
    for iteration in range(n_iter_max):
        tic = time.time()

        if gamma_line_search and iteration > 5: # No gamma line search for the first 5 iterations
            W, H, err, log_det, gamma, trace, cond_number = one_step_minvol_beta_nmf(data=data, W=W, H=H, rank=rank, beta=beta, delta=delta, lambda_=lambda_, gamma=gamma, tol_update_lagrangian=tol_update_lagrangian, prev_error=cost_fct_vals[-1])
        else:
            W, H, err, log_det, _, trace, cond_number = one_step_minvol_beta_nmf(data=data, W=W, H=H, rank=rank, beta=beta, delta=delta, lambda_=lambda_, gamma=None, tol_update_lagrangian=tol_update_lagrangian, prev_error=None)

        toc.append(time.time() - tic)

        cost_fct_vals.append(err)
        traceSave.append(trace)
        logdetSave.append(log_det)
        condNumberSave.append(cond_number)

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

def one_step_minvol_beta_nmf(data, W, H, rank, beta, delta, lambda_, gamma, tol_update_lagrangian, prev_error):
    assert beta == 1, "This function is only implemented for beta = 1 (KL divergence)."

    # Initialize W for gamma line search
    W_prev = W.copy() if gamma is not None else None

    H = mu.switch_alternate_mu(data, W, H, beta, "H") # np.transpose(multiplicative_updates.mu_betadivmin(H.T, W.T, data.T, beta))

    if gamma is not None:
        # Update W
        W_update, Y = min_vol_mu.KL_mu_min_vol(data=data, W=W, H=H, delta=delta, lambda_=lambda_, gamma = gamma, tol_update_lagrangian = tol_update_lagrangian)

        # Simplex projection wasn't working, so we turn to standard normlization instead (TODO: check avec Valentin si c'est bon)
        W_normalized, H_normalized = normalize_WH(W_update, H, "W")
        W, H, gamma = min_vol_mu.gamma_line_search(data, W_update = W_update, W_gamma_init = W_normalized, W_prev=W_prev, H_gamma_init = H_normalized, 
                                                   beta=beta, delta=delta, gamma_init=gamma, lambda_tilde=lambda_, prev_error=prev_error)
    else:
        W_update, Y = min_vol_mu.KL_mu_min_vol(data=data, W=W, H=H, delta=delta, lambda_=lambda_, gamma = None, tol_update_lagrangian = tol_update_lagrangian)

                
    # Compute the loss function
    log_det = min_vol_mu.compute_log_det(W, delta)
    err = beta_div.beta_divergence(data, W @ H, beta) + lambda_ * log_det

    trace = np.trace(Y @ (W.T @ W))
    cond_number = np.linalg.cond(W.T @ W + delta * np.eye(rank))

    return W, H, err, log_det, gamma, trace, cond_number

# def one_step_minvol_beta_nmf_lagrangian(data, W, H, rank, beta, delta, alpha, lambda_, tol_update_lagrangian):
#     assert beta == 1, "This function is only implemented for beta = 1 (KL divergence)."

#     H = multiplicative_updates.switch_alternate_mu(data, W, H, beta, "H") # np.transpose(multiplicative_updates.mu_betadivmin(H.T, W.T, data.T, beta))

#     W, Y = min_vol_mu.KL_mu_min_vol_lagrangian(data, W, H, delta, lambda_, tol_update_lagrangian)

#     # Compute the loss function
#     log_det = min_vol_mu.compute_log_det(W, delta)
#     err = beta_divergence(data, W @ H, beta) + lambda_ * log_det

#     trace = np.trace(Y @ (W.T @ W))
#     cond_number = np.linalg.cond(W.T @ W + delta * np.eye(rank))

#     return W, H, err, log_det, trace, cond_number

if __name__ == "__main__":
    np.random.seed(42)
    m, n, rank = 100, 200, 5
    W_0, H_0 = np.random.rand(m, rank), np.random.rand(rank, n) # Example input matrices
    data = W_0@H_0 + 1e-2*np.random.rand(m,n)  # Example input matrix
    
    W, H, lossfun, t = minvol_beta_nmf(data, rank, beta = 1, n_iter_max = 100, gamma_line_search = True, gamma = 1, verbose = False)
    print(f"Time for gamma line search: {t[-1]}")
    W, H, lossfun, t = minvol_beta_nmf(data, rank, beta = 1, n_iter_max = 100, gamma_line_search = False, verbose = False)
    print(f"Time without gamma line search, and with the normalization implemented inside the update rule: {t[-1]}")

    # TO DEBUG
    # W, H, lossfun, t = minvol_beta_nmf(data, rank, beta = 1, n_iter_max = 100, init = "nndsvd", gamma_line_search = True, gamma = 1, verbose = True)
    # W, H, lossfun, t = minvol_beta_nmf(data, rank, beta = 1, n_iter_max = 100, init = "nndsvd", gamma_line_search = False, verbose = True)
