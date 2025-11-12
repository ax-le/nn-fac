import numpy as np
import warnings

from nn_fac.nmf import nmf
from nn_fac.utils.normalize_wh import normalize_WH

def multilayer_beta_NMF(data, all_ranks, beta = 1, delta = 1e-6, n_iter_max_each_nmf = 100, init_each_nmf = "nndsvd", return_errors = False, verbose = False, deterministic=False, seed=0):
    if deterministic:
        np.random.seed(seed)

    # delta is useless here, because we use our own beta_nmf.
    L = len(all_ranks)
    assert L > 1, "The number of layers must be at least 2. Otherwise, ou should just use NMF"
    if min(data.shape) < max(all_ranks):
        count = 0
        min_data = min(data.shape)
        for idx, rank in enumerate(all_ranks):
            if min_data < rank:
                all_ranks[idx] = min_data
                count += 1
        print(f"The ranks are too high for the input matrix. The {count} larger ranks were set to {min_data} instead.")
        warnings.warn("Ranks have been changed.")
        
    if sorted(all_ranks, reverse=True) != all_ranks:
        raise ValueError("The ranks of deep NMF should be decreasing.")

    W = [None] * L
    H = [None] * L
    toc = [None] * L
    reconstruction_errors = np.empty((L, n_iter_max_each_nmf))
    reconstruction_errors.fill(None)

    W[0], H[0], reconstruction_errors[0], toc[0] = one_layer_update(data=data, rank=all_ranks[0], beta=beta, delta=delta, init_each_nmf=init_each_nmf, n_iter_max_each_nmf=n_iter_max_each_nmf, verbose=verbose, deterministic=deterministic, seed=seed)
    
    for i in range(1, L): # Layers
        W_i, H_i, errors_i, toc_i = one_layer_update(data=W[i - 1], rank=all_ranks[i], beta=beta, delta=delta, init_each_nmf=init_each_nmf, n_iter_max_each_nmf=n_iter_max_each_nmf, verbose=verbose, deterministic=deterministic, seed=seed)
        W[i], H[i], reconstruction_errors[i], toc[i] = W_i, H_i, errors_i, toc_i
        if verbose:
            print(f'Layer {i} done.')

    if return_errors:
        return W, H, reconstruction_errors, toc
    else:
        return W, H

def one_layer_update(data, rank, beta, delta, init_each_nmf, n_iter_max_each_nmf, verbose, deterministic=False, seed=0):
    W, H, cost_fct_vals, times = nmf(data, rank, init = init_each_nmf, U_0 = None, V_0 = None, n_iter_max=n_iter_max_each_nmf, tol=1e-8,
                                     update_rule = "mu", beta = beta,
                                     sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, True],
                                     verbose=verbose, return_costs=True, deterministic=deterministic, seed=seed)
    W_normalized, H_normalized = normalize_WH(W, H, matrix="H")
    reconstruction_errors = np.array(cost_fct_vals)
    toc = np.sum(times)
    return W_normalized, H_normalized, reconstruction_errors, toc

if __name__ == "__main__":
    np.random.seed(0)
    m, n, all_ranks = 100, 200, [15,10,5]
    data = np.random.rand(m, n)  # Example input matrix
    W, H, reconstruction_errors, toc = multilayer_beta_NMF(data, all_ranks, n_iter_max_each_nmf = 100, verbose = True)
    print(f"Losses: {reconstruction_errors}")
