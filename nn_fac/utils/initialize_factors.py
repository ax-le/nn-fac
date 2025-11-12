"""
A file to initialize factor matrices for matrix/tensor factorization models.

Essentially, it comprises random and nndsvd initializations.

nndsvd initialization is based on [1], and was initially implemented in the nimfa package [2]. Due to problems in maintenance of nimfa, the relevant code has been included here. Not that is it mostly a copy-paste job with minor modifications (for maintenance).
Only the core nndsvd function has been extracted here, and without sparse matrices acceleration options.

Here is a description of nndsvd from nimfa's nndsvd.py:
    Nonnegative Double Singular Value Decomposition (NNDSVD) [Boutsidis2007]_ is a
    new method designed to enhance the initialization stage of the nonnegative
    matrix factorization. The basic algorithm contains no randomization and is
    based on two SVD processes, one approximating the data matrix, the other
    approximating positive sections of the resulting partial SVD factors utilizing
    an algebraic property of unit rank matrices.

    NNDSVD is well suited to initialize NMF algorithms with sparse factors.
    Numerical examples suggest that NNDSVD leads to rapid reduction of the
    approximation error of many NMF algorithms.
    
[1]: C. Boutsidis and E. Gallopoulos. "SVD based initialization: A head start for nonnegative matrix factorization,"Pattern Recognition 41.4 (2008), pp. 1350-1362.
[2]: B. Zupan et al. "Nimfa: A python library for nonnegative matrix factorization", Journal of Machine Learning Research 13.Mar (2012), pp. 849-853.
"""

from unittest import case
import warnings
import numpy as np
import random
import tensorly as tl
from tensorly.decomposition import tucker as tl_tucker

import nn_fac.utils.errors as err

# %% Main initialization functions
def nmf_initialization(data, rank, init_type, deterministic = False, seed = 0):
    match init_type.lower():
        case "nndsvd":
            return nndsvd(data, rank)
        case "random":
            if deterministic:
                np.random.seed(seed)
                random.seed(seed)
            m, n = data.shape
            U_0 = np.random.rand(m, rank)
            V_0 = np.random.rand(rank, n)
            return U_0, V_0
        case _:
            raise err.InvalidInitializationType("Initialization type not understood.")
        
def ntd_initialization(tensor, ranks, init_type, deterministic = False, seed = 0):
    nb_modes = len(tensor.shape)
    match init_type.lower():
        case "random":
            factors = []
            if deterministic:
                np.random.seed(seed)
                random.seed(seed)
            for mode in range(nb_modes):
                random_array = np.random.rand(tensor.shape[mode], ranks[mode])
                one_factor = tl.tensor(random_array)
                one_factor[one_factor < 1e-12] = 1e-12 # To avoid zeros
                factors.append(one_factor)
            the_core = np.random.rand(np.prod(ranks)).reshape(tuple(ranks))
            the_core[the_core < 1e-12] = 1e-12 # To avoid zeros
            core = tl.tensor(the_core)
            return core, factors

        case "tucker":        
            if deterministic:
                init_core, init_factors = tl_tucker(tensor, ranks, random_state = seed)
            else:
                init_core, init_factors = tl_tucker(tensor, ranks)
            factors = [tl.abs(f) + 1e-12 for f in init_factors]
            core = tl.abs(init_core) + 1e-12
            return core, factors

        case "chromas": # Tucker where W is fixed to I12
            core, factors = ntd_initialization(tensor, ranks, init_type="tucker", deterministic=deterministic, seed=seed)
            factors[0] = np.identity(12) # Hardcoded
            return core, factors

        case _:
            raise err.InvalidInitializationType("Initialization type not understood.")
        
def ntf_initialization(tensor, rank, init_type, deterministic = False, seed = 0):
    nb_modes = len(tensor.shape)
    factors = []

    if deterministic:
        np.random.seed(seed)
        random.seed(seed)
    
    match init_type.lower():
        case "random":
            for mode in range(nb_modes):
                factors.append(tl.tensor(np.random.rand(tensor.shape[mode], rank)))
            return factors

        case "nndsvd":
            for mode in range(nb_modes):
                if tensor.shape[mode] < rank: # If the mode is smaller than the rank, fall back to random initialization
                    current_factor = np.random.rand(tensor.shape[mode], rank)
                else: # Otherwise, use nndsvd on the unfolded tensor
                    current_factor, _ = nndsvd(tl.unfold(tensor, mode), rank)
                factors.append(tl.tensor(current_factor))
            return factors

        case _:
            raise err.InvalidInitializationType("Initialization type not understood.")
        
def parafac2_initialization(tensor_slices, rank, init_type, init_with_P, deterministic = False, seed = 0):
    nb_channel = len(tensor_slices)
    r, n = tensor_slices[0].shape
    W_list = []
    D_list = []

    if deterministic:
        np.random.seed(seed)
        random.seed(seed)

    match init_type.lower():
        case "random":
            H = np.random.rand(rank, n)
            for k in range(nb_channel):
                W_list.append(np.random.rand(r, rank))
                D_list.append(np.diag(np.random.rand(rank)))
            D_list = np.array(D_list)
            if init_with_P:
                zero_padded_identity = np.identity(r)
                zero_padded_identity = zero_padded_identity[:,0:rank]
                P_list = [zero_padded_identity for i in range(nb_channel)]
                W_star = None
            else:
                W_star = np.random.rand(r, rank)
                P_list = None
            
            return W_list, H, D_list, P_list, W_star

        case "nndsvd":
            for k in range(nb_channel):
                W_k, H = nndsvd(tensor_slices[k], rank)
                W_list.append(W_k)
                D_list.append(np.diag(np.random.rand(rank)))
            D_list = np.array(D_list)

            if init_with_P:
                zero_padded_identity = np.identity(r)
                zero_padded_identity = zero_padded_identity[:,0:rank]
                P_list = [zero_padded_identity for i in range(nb_channel)]
                W_star = None
            else:
                W_star_local = np.zeros(W_list[0].shape)
                for k in range(nb_channel):
                    W_star_local += W_list[k]
                W_star = np.divide(W_star_local, k)
                P_list = None


# %% Common init methods
def nndsvd(V, rank):
    def _is_negative(V):
        if V.any() < 0:
            return True
        
    def _pos(X):
        return np.multiply(X >= 0, X)

    def _neg(X):
        return np.multiply(X < 0, -X)

    if _is_negative(V):
        raise ValueError("The input matrix contains negative elements.")
    U, S, E = np.linalg.svd(V)
    E = E.T

    W = np.zeros((V.shape[0], rank))
    H = np.zeros((rank, V.shape[1]))
    # choose the first singular triplet to be nonnegative
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(E[:, 0].T)
    # second svd for the other factors
    for i in range(1, rank):
        uu = U[:, i]
        vv = E[:, i]
        uup = _pos(uu)
        uun = _neg(uu)
        vvp = _pos(vv)
        vvn = _neg(vv)
        n_uup = np.linalg.norm(uup, 2)
        n_vvp = np.linalg.norm(vvp, 2)
        n_uun = np.linalg.norm(uun, 2)
        n_vvn = np.linalg.norm(vvn, 2)
        termp = n_uup * n_vvp
        termn = n_uun * n_vvn
        if (termp >= termn):
            W[:, i] = np.sqrt(S[i] * termp) / n_uup * uup
            H[i, :] = np.sqrt(S[i] * termp) / n_vvp * vvp.T
        else:
            W[:, i] = np.sqrt(S[i] * termn) / n_uun * uun
            H[i, :] = np.sqrt(S[i] * termn) / n_vvn * vvn.T
    # Important, not to be stuck on zeroes
    # W[W < 1e-12] = 1e-12 
    # H[H < 1e-12] = 1e-12
    W = np.maximum(W, 1e-12)
    H = np.maximum(H, 1e-12)
    return W, H