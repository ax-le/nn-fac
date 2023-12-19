##### SIMPLEX NMF
## Projects H on the unit simplex, comes from 'Leplat, V., Gillis, N., & Idier, J. (2021). Multiplicative updates for NMF with β-divergences under disjoint equality constraints. SIAM Journal on Matrix Analysis and Applications, 42(2), 730-752. arXiv:2010.16223.'
## Will be integrated into nmf.py when stable.

import numpy as np
import time
from nimfa.methods import seeding

import nn_fac.update_rules.mu as mu
import nn_fac.utils.beta_divergence as beta_div
import nn_fac.utils.errors as err

eps = 1e-12 # np.finfo(np.float64).eps

def simplex_beta_nmf(data, rank, beta, n_iter_max = 100, tol = 1e-8, tol_update_lagrangian = 1e-6, init = "random", W_0 = None, H_0 = None, verbose = False):  
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
    
    return compute_simplex_beta_nmf(data=data, W_0=W, H_0=H, rank=rank, beta=beta,n_iter_max=n_iter_max, tol=tol, tol_update_lagrangian=tol_update_lagrangian, verbose = verbose)

def compute_simplex_beta_nmf(data, W_0, H_0, rank, beta, n_iter_max = 100, tol = 1e-8, tol_update_lagrangian = 1e-6, verbose = False):
    W, H = W_0.copy(), H_0.copy()

    # Array to save the value of the loss function
    cost_fct_vals = []
    toc = []

    # Optimization loop
    for iteration in range(n_iter_max):
        tic = time.time()

        W, H, err = one_step_simplex_beta_nmf(data=data, W=W, H=H, rank=rank, beta=beta, tol_update_lagrangian=tol_update_lagrangian)

        toc.append(time.time() - tic)
        cost_fct_vals.append(err)

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

def one_step_simplex_beta_nmf(data, W, H, rank, beta, tol_update_lagrangian):
    W = mu.switch_alternate_mu(data, W, H, beta, "W")
    H = mu.simplex_proj_mu(data, W, H, beta, tol_update_lagrangian)
    err = beta_div.beta_divergence(data, W@H, beta)
    return W, H, err

if __name__ == "__main__":
    np.random.seed(42)
    m, n, rank = 100, 200, 5
    W_0, H_0 = np.random.rand(m, rank), np.random.rand(rank, n) # Example input matrices
    data = W_0@H_0 + 1e-2*np.random.rand(m,n)  # Example input matrix
    
    W, H, lossfun, t = simplex_beta_nmf(data, rank, beta = 1, n_iter_max = 100, verbose = True)

    # TO DEBUG
    # W, H, lossfun, t = minvol_beta_nmf(data, rank, beta = 1, n_iter_max = 100, init = "nndsvd", gamma_line_search = True, gamma = 1, verbose = True)
    # W, H, lossfun, t = minvol_beta_nmf(data, rank, beta = 1, n_iter_max = 100, init = "nndsvd", gamma_line_search = False, verbose = True)
