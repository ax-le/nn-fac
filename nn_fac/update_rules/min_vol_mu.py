import warnings
import numpy as np

import nn_fac.utils.normalize_wh as normalize_wh
from nn_fac.utils.beta_divergence import beta_divergence

eps = 1e-12

def KL_mu_min_vol(data, W, H, delta, lambda_, gamma = None, tol_update_lagrangian = 1e-6):
    m, n = data.shape
    k = W.shape[1]
    Jm1 = np.ones((m,1))
    ONES = np.ones((m, n))

    # Compute Y
    Y = compute_Y(W, delta)

    # Update W
    Y_plus = np.maximum(0, Y)
    Y_minus = np.maximum(0, -Y)
    C = ONES @ H.T - 4 * lambda_ * W @ Y_minus
    S = 8 * lambda_ * (W @ (Y_plus + Y_minus)) * ((data / ((W @ H) + eps)) @ H.T)
    D = 4 * lambda_ * W @ (Y_plus + Y_minus)

    if gamma is not None:
        W = W * ((C ** 2 + S) ** 0.5 - C) / (D + eps)

    else:
        lagragian_multipliers_0 = np.zeros((k, 1)) #(D[:,0] - C[:,0] * W[:,0]).T
        lagragian_multipliers = update_lagragian_multipliers_Wminvol(C, S, D, W, lagragian_multipliers_0, tol_update_lagrangian)

        W = W * ((((C + Jm1 @ lagragian_multipliers.T) ** 2 + S) ** 0.5 - (C + Jm1 @ lagragian_multipliers.T)) / (D + eps))
        
    W = np.maximum(W, eps)

    return W, Y

def gamma_line_search(data, W_update, W_gamma_init, H_gamma_init, beta, delta, gamma_init, lambda_tilde, W_prev, prev_error):
    W_gamma = W_gamma_init.copy()
    H_gamma = H_gamma_init.copy()
    gamma = gamma_init
    cur_log_det = compute_log_det(W_gamma, delta)
    cur_err = beta_divergence(data, W_gamma @ H_gamma, beta) + lambda_tilde * cur_log_det
    while cur_err > prev_error and gamma > 1e-16:
        gamma *= 0.8
        W_gamma = (1 - gamma) * W_prev + gamma * W_update
        W_gamma, H_gamma = normalize_wh.normalize_WH(W_gamma, H_gamma, "W")
        cur_log_det = compute_log_det(W_gamma, delta)
        cur_err = beta_divergence(data, W_gamma @ H_gamma, beta) + lambda_tilde * cur_log_det
    
    gamma = min(gamma * 1.2, 1)
    return W_gamma, H_gamma, gamma

def update_lagragian_multipliers_Wminvol(C, S, D, W, lagrangian_multipliers_0, tol = 1e-6, n_iter_max = 100):
    # Comes from Multiplicative Updates for NMF with β-Divergences under Disjoint Equality Constraints, https://arxiv.org/pdf/2010.16223.pdf
    m, k = W.shape
    Jm1 = np.ones((m,1))
    Jk1 = np.ones(k)
    ONES = np.ones((m, k))
    lagrangian_multipliers = lagrangian_multipliers_0.copy()

    for iter in range(n_iter_max):
        lagrangian_multipliers_prev = lagrangian_multipliers.copy()
        Mat = W * ((((C + Jm1 @ lagrangian_multipliers.T) ** 2 + S) ** 0.5) - (C + Jm1 @ lagrangian_multipliers.T)) / (D + eps)
        Matp = W * ((((C + Jm1 @ lagrangian_multipliers.T) ** 2 + S) **(-0.5)) - ONES) / (D + eps)
        # Matp = (W / (D + eps)) * ((C + Jm1 @ lagrangian_multipliers.T) / (((C + Jm1 @ lagrangian_multipliers.T)**2 + S)**0.5) - ONES) # Was also in the code, may be more efficient due to less computation of matrix power.


        xi = np.sum(Mat, axis=0) - Jk1
        xip = np.sum(Matp, axis=0)
        lagrangian_multipliers = lagrangian_multipliers - (xi / xip).reshape((k,1))

        if np.max(np.abs(lagrangian_multipliers - lagrangian_multipliers_prev)) <= tol:
            break

        if iter == n_iter_max - 1:
            warnings.warn('Maximum of iterations reached in the update of the Lagrangian multipliers.')

    return lagrangian_multipliers

def compute_Y(W, delta):
    r = W.shape[1]
    return np.linalg.inv((W.T @ W + delta * np.eye(r)))# + eps)

def compute_det(W, delta):
    r = W.shape[1]
    det = np.linalg.det(W.T @ W + delta * np.eye(r))
    #det += eps ## TODO: Demander aussi à Valentin si c'est ok ce eps sur le det.
    return det

def compute_log_det(W, delta):
    det = compute_det(W, delta)
    return np.log10(det, where=(det!=0))


# def KL_mu_min_vol_lagrangian(data, W, H, delta, lambda_, tol_update_lagrangian):
#     m, n = data.shape
#     k = W.shape[1]
#     Jm1 = np.ones((m,1))
#     ONES = np.ones((m, n))

#     # Compute Y
#     Y = compute_Y(W, delta)

#     # Update mu (lagrangian multipliers)
#     Y_plus = np.maximum(0, Y)
#     Y_minus = np.maximum(0, -Y)
#     C = ONES @ H.T - 4 * lambda_ * W @ Y_minus
#     S = 8 * lambda_ * (W @ (Y_plus + Y_minus)) * ((data / ((W @ H) + eps)) @ H.T)
#     D = 4 * lambda_ * W @ (Y_plus + Y_minus)

#     lagragian_multipliers_0 = np.zeros((k, 1)) #(D[:,0] - C[:,0] * W[:,0]).T
#     lagragian_multipliers = update_lagragian_multipliers(C, S, D, W, lagragian_multipliers_0, tol_update_lagrangian)

#     # Update W
#     W = W * ((((C + Jm1 @ lagragian_multipliers.T) ** 2 + S) ** 0.5 - (C + Jm1 @ lagragian_multipliers.T)) / (D + eps))
#     W = np.maximum(W, eps)

#     return W, Y