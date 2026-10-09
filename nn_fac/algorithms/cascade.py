import nn_fac.nmf as nnfacnmf
import nn_fac.hnmf as nnfachnmf
import numpy as np
from tqdm import tqdm
import random
import time
import random

"""
Cascade algorithms, for now implemented for NMF and HNMF. Learns representations matrices W, or S and A over a list of data matrices, enabling for
an inference phase where an H matrix is optimized for every data seen.
"""

def nmf_cascade(Xs, rank, betas, update_rules, n_iters, n_stepW=100, n_stepH=100, W_init=0, n_epoch=100, 
                verbose=False, return_costs=False, update_order="HW", sparsity_coefficients=[0.0, 0.0], fixed_modes=[], normalize=[True, False], randomize_ord=True, deterministic=True):
    """
    Parameters :
    ------------
        Xs : a list of arrays for the successive spectrograms.
        rank : int for the rank in nmf computations.
        betas : list of beta values for the beta divergences used to compute the successive nmfs. If given a single beta : same is applied for all nmfs.
        update_rules : list of strings for successive update rules used. If given a single rule : same is applied for all nmfs.
        n_iters : list of int for successive number of iterations in nmf computation. If given a single value : same is applied for all nmfs.
        n_stepW, n_stepH : ints for the numbers of steps in HALS updates if the update rule used is HALS.
        spar_c : float indicating the sparsity coefficient in the update of H.
        W_init : Either 0 if W is initialised randomly, or a starting note dictionnary.
        n_epoch : int for the number of epoch of the learning phase for W.
        verbose : boolean used to indicate if the details of the nmfs computations are shown or not.
        return_costs : boolean indicating if the costs of reconstruction from the successive NMFs are returned.
        update_order : string indicating the update order of the matrices W and H at each iteration of the NMF.
        randomize_ord : boolean indicating if the order in which the spectrograms are viewed in the training phase is randomized or not.
        deterministic : boolean indicating if the hnmf is deterministic or not.
    
    Returns :
    ---------
        Ws,Hs : list of arrays such that np.dot(Wi,Hi) approximates Vi. Ws[-1] contains the final notes dictionnary shaped through multiple spectrograms. 
        t : float indicating the time of the learning phase.
    """
    n_nmf = len(Xs)
    Ws = []
    Hs = []
    Deltas = []

    if type(betas) == int:
        betas = [betas for _ in range(n_nmf)] 
    if type(update_rules) == str:
        update_rules = [update_rules for _ in range(n_nmf)] 
    if type(n_iters) == int:
        n_iters = [n_iters for _ in range(n_nmf)] 

    t = time.time()
    costs = [[] for _ in range(n_nmf)]

    for epoch in tqdm(range(1, n_epoch+1)):
        ind = range(n_nmf)
        if randomize_ord:
            ind = [i for i in range(n_nmf)]
            random.shuffle(ind)
        
        for step in tqdm(ind):
            X = Xs[step]
            Hinit = np.random.rand(rank, len(X[0,:])) + 1e-12
            #TODO : allow to use something else than random for H only, W being derived from past iterations ?
            if Ws == []:
                if type(W_init) is int:
                    Winit = np.random.rand(len(X[:,0]), rank) + 1e-12
                else :
                    Winit = W_init.copy()
            else:
                Winit = Ws[-1].copy()

            if return_costs:
                W, H, c, _ = nnfacnmf.nmf(Xs[step], rank, init="custom", U_0=Winit, V_0=Hinit, n_iter_max=n_iters[step], n_stepU=n_stepW, n_stepV=n_stepH, tol=1e-6, update_rule=update_rules[step],
                                    beta=betas[step], sparsity_coefficients=sparsity_coefficients, fixed_modes=fixed_modes, normalize=normalize, update_order=update_order, verbose=verbose, return_costs=True, deterministic=deterministic)
                costs[step].append(c)
                Ws.append(W)
                Deltas.append(np.linalg.norm(Winit-W))
                Hs.append(H)
            else:
                W, H = nnfacnmf.nmf(Xs[step], rank, init="custom", U_0=Winit, V_0=Hinit, n_iter_max=n_iters[step], n_stepU=n_stepW, n_stepV=n_stepH, tol=1e-6, update_rule=update_rules[step],
                                    beta=betas[step], sparsity_coefficients=sparsity_coefficients, fixed_modes=fixed_modes, normalize=normalize, update_order=update_order, verbose=verbose, return_costs=False, deterministic=deterministic)
                Ws.append(W)
                Hs.append(H)

    t = time.time()-t

    for step in range(n_nmf):
        flat_costs = sum(costs[step], [])
        costs[step] = flat_costs

    if return_costs:
        return Ws, Hs, costs, t
    
    return Ws,Hs,t

def hnmf_cascade(Xs, E, rank, betas, update_rules, n_iters, n_stepS=100, n_stepH=100, S_init=0, n_epoch=100, 
                verbose=False, return_costs=False, update_order="HS", sparsity_coefficients=[0.0, 0.0, 0.0], fixed_modes=[0], normalize=[True, False, False], randomize_ord=True, deterministic=True):
    """
    Parameters :
    ------------
        Xs : a list of arrays for the successive spectrograms.
        E : the harmonic relations matrix used in the HNMF factorization.
        rank : int for the rank in hnmf computations.
        betas : list of beta values for the beta divergences used to compute the successive nmfs. If given a single beta : same is applied for all hnmfs.
        update_rules : list of strings for successive update rules used. If given a single rule : same is applied for all hnmfs.
        n_iters : list of int for successive number of iterations in hnmf computation. If given a single value : same is applied for all hnmfs.
        n_stepS, n_stepH : ints for the numbers of steps in HALS updates if the update rule used is HALS.
        spar_c : float indicating the sparsity coefficient in the update of H.
        S_init : Either 0 if S is initialised randomly, or a starting fundamentals dictionnary.
        n_epoch : int for the number of epoch of the learning phase for W.
        verbose : boolean used to indicate if the details of the nmfs computations are shown or not.
        return_costs : boolean indicating if the costs of reconstruction from the successive HNMFs are returned.
        update_order : string indicating the update order of the matrices S and H at each iteration of the HNMF.
        randomize_ord : boolean indicating if the order in which the spectrograms are viewed in the training phase is randomized or not.
        deterministic : boolean indicating if the hnmf is deterministic or not.
    
    Returns :
    ---------
        Ss, Hs, As : list of arrays such that np.dot(np.dot(Si,np.multiply(E,Ai)),Hi) approximates Vi. Ss[-1] contains the final fundamentals dictionnary shaped through multiple spectrograms. 
        t : float indicating the time of the learning phase.
    """
    n_nmf = len(Xs)
    Ss = []
    As = []
    Hs = []
    Deltas = []

    if type(betas) == int:
        betas = [betas for _ in range(n_nmf)] 
    if type(update_rules) == str:
        update_rules = [update_rules for _ in range(n_nmf)] 
    if type(n_iters) == int:
        n_iters = [n_iters for _ in range(n_nmf)] 

    t = time.time()
    costs = [[] for _ in range(n_nmf)]

    for epoch in tqdm(range(1, n_epoch+1)):
        ind = range(n_nmf)
        if randomize_ord:
            ind = [i for i in range(n_nmf)]
            random.shuffle(ind)
        
        for step in tqdm(ind):
            X = Xs[step]
            Hinit = np.random.rand(rank, len(X[0,:])) + 1e-12
            #TODO : allow to use something else than random for H only, W being derived from past iterations ?
            if Ss == []:
                if type(S_init) is int:
                    Sinit = np.random.rand(len(X[:,0]), rank) + 1e-12
                else :
                    Sinit = S_init.copy()
                Ainit = np.ones((rank,rank))#np.random.rand(rank,rank) + 1e-12
            else:
                Sinit = Ss[-1].copy()
                Ainit = As[-1].copy()

            if return_costs:
                S, H, A, c, _ = nnfachnmf.hnmf(Xs[step], rank, E, init="custom", S_0=Sinit, A_0=Ainit, V_0=Hinit, n_iter_max=n_iters[step], n_stepS=n_stepS, n_stepV=n_stepH, tol=1e-6, update_rule=update_rules[step],
                                    beta=betas[step], sparsity_coefficients=sparsity_coefficients, fixed_modes=fixed_modes, normalize=normalize, update_order=update_order, verbose=verbose, return_costs=True, deterministic=deterministic)
                costs[step].append(c)
                Ss.append(S)
                As.append(A)
                #Deltas.append(np.linalg.norm(Sinit-S))
                Hs.append(H)
            else:
                S, H, A = nnfachnmf.hnmf(Xs[step], rank, E, init="custom", S_0=Sinit, A_0=Ainit, V_0=Hinit, n_iter_max=n_iters[step], n_stepS=n_stepS, n_stepV=n_stepH, tol=1e-6, update_rule=update_rules[step],
                                    beta=betas[step], sparsity_coefficients=sparsity_coefficients, fixed_modes=fixed_modes, normalize=normalize, update_order=update_order, verbose=verbose, return_costs=False, deterministic=deterministic)
                Ss.append(S)
                As.append(A)
                Hs.append(H)

    t = time.time()-t

    for step in range(n_nmf):
        flat_costs = sum(costs[step], [])
        costs[step] = flat_costs

    if return_costs:
        return Ss, Hs, As, costs, t
    
    return Ss, Hs, As, t