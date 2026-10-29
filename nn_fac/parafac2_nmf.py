import numpy as np
import tensorly as tl
import nn_fac.algorithms.cascade as csc

def parafac2_nmf(Xs, rank, betas, update_rules, n_iters, n_stepW=100, n_stepH=100, Ws_init=None, n_epoch=100, 
                verbose=False, update_order="HW", sparsity_coefficients=[0.0, 0.0], fixed_modes=[0,0], normalize=[True, False], randomize_ord=True, deterministic=True):
    """
    Obtaining a PARAFAC2 like decomposition by cascading the C^T matrix through multiple NMFs, one for each layer of matrices.

    Xs : list of matrices
    """

    Xs_T = [tl.transpose(X) for X in Xs]

    if update_order == "HW":
        up_ord = "WH"
    else :
        up_ord = "HW"
    # Reversed copies (not in-place) : index 0 now refers to H^T, index 1 to W^T
    sparsity_coefficients = sparsity_coefficients[::-1]
    fixed_modes = fixed_modes[::-1]
    normalize = normalize[::-1]

    Cs, _, tc =csc.nmf_cascade_training(Xs_T, rank, betas, update_rules, n_iters, n_stepW=n_stepH, n_stepH=n_stepW, W_init=0, n_epoch=n_epoch,
                                        verbose=verbose, return_costs=False, update_order=up_ord, sparsity_coefficients=sparsity_coefficients,
                                        fixed_modes=fixed_modes, normalize=normalize, randomize_ord=randomize_ord, deterministic=deterministic)

    C = Cs[-1]
    Ws_T_init = None if Ws_init is None else [tl.transpose(W) for W in Ws_init]

    Ws_T, ti = csc.nmf_cascade_testing(Xs_T, C, rank, betas, update_rules, n_iters, n_stepH=n_stepW, Hs_init=Ws_T_init, verbose=verbose,
                                        return_costs=False, sparsity_coefficient=sparsity_coefficients[1], normalize=normalize[1], deterministic=deterministic)

    Ws = [tl.transpose(W) for W in Ws_T]
    H = tl.transpose(C)
    t = tc + ti

    return Ws, H, t