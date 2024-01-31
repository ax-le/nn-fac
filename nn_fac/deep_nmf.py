import numpy as np
import warnings
import time

import nn_fac.update_rules.mu as mu
import nn_fac.update_rules.deep_mu as deep_mu

import nn_fac.multilayer_nmf as multi_nmf

import nn_fac.utils.beta_divergence as beta_div
from nn_fac.utils.normalize_wh import normalize_WH

def deep_KL_NMF(data, all_ranks, n_iter_max_each_nmf = 100, n_iter_max_deep_loop = 100, init = "multilayer_nmf", init_multi_layer = "random", W_0 = None, H_0 = None, delta = 1e-6, tol = 1e-6, return_errors = False, verbose = False):
    L = len(all_ranks)

    assert L > 1, "The number of layers must be at least 2. Otherwise, ou should just use NMF."
    all_errors = np.zeros((n_iter_max_deep_loop + 1,L))
    toc = []
    global_errors = []

    if sorted(all_ranks, reverse=True) != all_ranks:
        raise ValueError("The ranks of deep NMF should be decreasing.")
        #warnings.warn("Warning: The ranks of deep NMF should be decreasing.")

    if init == "multilayer_nmf":
        W, H, e = multi_nmf.multilayer_beta_NMF(data, all_ranks, n_iter_max_each_nmf = n_iter_max_each_nmf, init_each_nmf = init_multi_layer, delta = delta, return_errors = True, verbose = False)
        all_errors[0] = e

    elif init == "custom":
        W = W_0
        H = H_0
        all_errors[0,0] = beta_div.kl_divergence(data, W[0] @ H[0])
        for i in range(1,L):
            all_errors[0,i] = [beta_div.kl_divergence(W[i-1], W[i] @ H[i])]

    lambda_ = 1 / np.array(all_errors[0])

    global_errors.append(lambda_.T @ all_errors[0])

    for deep_iteration in range(n_iter_max_deep_loop):
        tic = time.time()

        W, H, errors = one_step_deep_KL_nmf(data, W, H, all_ranks, lambda_, delta)

        toc.append(time.time() - tic)

        all_errors[deep_iteration + 1] = errors
        global_errors.append(lambda_.T @ errors)

        if verbose:

            if global_errors[-2] - global_errors[-1] > 0:
                print(f'Normalized sum of errors through layers={global_errors[-1]}, variation={global_errors[-2] - global_errors[-1]}.')
            else:
                # print in red when the reconstruction error is negative (shouldn't happen)
                print(f'\033[91m Normalized sum of errors through layers={global_errors[-1]}, variation={global_errors[-2] - global_errors[-1]}. \033[0m')

        if deep_iteration > 1 and abs(global_errors[-2] - global_errors[-1]) < tol:
            # Stop condition: relative error between last two iterations < tol
            if verbose:
                print(f'Converged in {deep_iteration} iterations.')
            break
    
    if return_errors:
        return W, H, all_errors, toc
    else:
        return W, H

def one_step_deep_KL_nmf(data, W, H, all_ranks, lambda_, delta):
    # delta is useless here, because we use our own beta_nmf.
    L = len(all_ranks)
    errors = []

    for layer in range(L):
        if layer == 0:
            lam = lambda_[1] / lambda_[0]
            H[0] = mu.switch_alternate_mu(data, W[0], H[0], beta=1, matrix="H")
            W[0], H[0] = normalize_WH(W[0], H[0], matrix="H")
            # H[0] = mu.simplex_proj_mu(data, W[0], H[0], beta=1)
            W[0] = deep_mu.deep_KL_mu(data, W[0], H[0], W[1] @ H[1], lam)
            errors.append(beta_div.kl_divergence(data, W[0] @ H[0]))

        elif layer == L - 1:
            H[layer] = mu.switch_alternate_mu(W[layer-1], W[layer], H[layer], beta=1, matrix="H")
            W[layer], H[layer] = normalize_WH(W[layer], H[layer], matrix="H")
            # H[layer] = mu.simplex_proj_mu(W[layer-1], W[layer], H[layer], beta=1)
            W[layer] = mu.switch_alternate_mu(W[layer-1], W[layer], H[layer], beta=1, matrix="W")
            errors.append(beta_div.kl_divergence(W[layer-1], W[layer] @ H[layer]))

        else:
            lam = lambda_[layer + 1] / lambda_[layer]
            H[layer] = mu.switch_alternate_mu(W[layer-1], W[layer], H[layer], beta=1, matrix="H")
            W[layer], H[layer] = normalize_WH(W[layer], H[layer], matrix="H")
            # H[layer] = mu.simplex_proj_mu(W[layer-1], W[layer], H[layer], beta=1)
            W[layer] = deep_mu.deep_KL_mu(W[layer-1], W[layer], H[layer], W[layer+1] @ H[layer+1], lam)
            errors.append(beta_div.kl_divergence(W[layer-1], W[layer] @ H[layer]))

    return W, H, errors

if __name__ == "__main__":
    np.random.seed(0)
    m, n, all_ranks = 100, 200, [15,10,5]
    data = np.random.rand(m, n)  # Example input matrix
    W, H, reconstruction_errors, toc = deep_KL_NMF(data, all_ranks, n_iter_max_each_nmf = 100, verbose = True)
