import nn_fac.nmf as nnfacnmf
import nn_fac.hnmf as nnfachnmf
import matcouply as mc
import numpy as np
import tensorly as tl
from tqdm import tqdm
import random
import time
import random

"""
Cascade algorithms, for now implemented for NMF and HNMF. Learns representations matrices W, or S and A over a list of data matrices (_training), enabling for
an inference phase (_testing) where an H matrix is optimized for every data seen while keeping W or SA fixed from the learning procedure. The testing procedure
is shared for both NMF and HNMF traing procedures since the dictionnary W or S(E°A) is fixed either way.
"""

""" ----------TRAINING PROCEDURES---------- """

def nmf_cascade_training(Xs, rank, betas, update_rules, n_iters, n_stepW=100, n_stepH=100, W_init=0, n_epoch=100, 
                verbose=False, return_costs=False, update_order="HW", sparsity_coefficients=[0.0, 0.0], fixed_modes=[0,0], normalize=[True, False], randomize_ord=True, deterministic=True):
    """
    Parameters :
    ------------
        Xs : a list of arrays for the successive spectrograms.
        rank : int for the rank in nmf computations.
        betas : list of beta values for the beta divergences used to compute the successive nmfs. If given a single beta : same is applied for all nmfs.
        update_rules : list of strings for successive update rules used. If given a single rule : same is applied for all nmfs.
        n_iters : list of int for successive number of iterations in nmf computation. If given a single value : same is applied for all nmfs.
        n_stepW, n_stepH : ints for the numbers of steps in HALS updates if the update rule used is HALS.
        W_init : Either 0 if W is initialised randomly, or a starting note dictionnary.
        n_epoch : int for the number of epoch of the learning phase for W.
        verbose : boolean used to indicate if the details of the nmfs computations are shown or not.
        return_costs : boolean indicating if the costs of reconstruction from the successive NMFs are returned.
        update_order : string indicating the update order of the matrices W and H at each iteration of the NMF.
        sparsity_coefficients : The sparsity coefficients for the different matrices updated through the NMFs.
        fixed_modes : List of ints indicating which matrices are updated or not throughout the NMFs.
        normalize : List of booleans indicating if the matrices are normalized or not during the NMFs.
        randomize_ord : boolean indicating if the order in which the spectrograms are viewed in the training phase is randomized or not.
        deterministic : boolean indicating if the nmfs are deterministics or not.
    
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

def hnmf_cascade_training(Xs, E, rank, betas, update_rules, n_iters, n_stepS=100, n_stepH=100, S_init=0, n_epoch=100, 
                verbose=False, return_costs=False, update_order="HS", sparsity_coefficients=[0.0, 0.0, 0.0], fixed_modes=[1,0,0], normalize=[True, False, False], randomize_ord=True, deterministic=True):
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
        S_init : Either 0 if S is initialised randomly, or a starting fundamentals dictionnary.
        n_epoch : int for the number of epoch of the learning phase for W.
        verbose : boolean used to indicate if the details of the nmfs computations are shown or not.
        return_costs : boolean indicating if the costs of reconstruction from the successive HNMFs are returned.
        update_order : string indicating the update order of the matrices S and H at each iteration of the HNMF.
        sparsity_coefficients : The sparsity coefficients for the different matrices updated through the HNMFs.
        fixed_modes : List of ints indicating which matrices are updated or not throughout the HNMFs.
        normalize : List of booleans indicating if the matrices are normalized or not during the HNMFs.
        randomize_ord : boolean indicating if the order in which the spectrograms are viewed in the training phase is randomized or not.
        deterministic : boolean indicating if the hnmfs are deterministics or not.
    
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


""" ---------TESTING PROCEDURES---------- """

def nmf_cascade_testing(Xs, W_casc, rank, betas, update_rules, n_iters,  n_stepH=100, Hs_init=None,
                verbose=False, return_costs=False, sparsity_coefficient=0.0, normalize=False, deterministic=True):
    """
    Parameters :
    ------------
        Xs : a list of arrays for the successive spectrograms.
        W_casc : the dictionnay obtained in the training process : Ws[-1] if the nmf_cascade_training procedure was used.
        rank : int for the rank in hnmf computations.
        betas : list of beta values for the beta divergences used to compute the successive nmfs. If given a single beta : same is applied for all nmfs.
        update_rules : list of strings for successive update rules used. If given a single rule : same is applied for all nmfs.
        n_iters : list of int for successive number of iterations in nmf computation. If given a single value : same is applied for all nmfs.
        n_stepH : int for the numbers of steps in HALS updates if the update rule used is HALS.
        Hs_init : The list of H arrays used for nmf initializations. If None, the initializations are random.
        verbose : boolean used to indicate if the details of the nmfs computations are shown or not.
        return_costs : boolean indicating if the costs of reconstruction from the NMFs are returned.
        sparsity_coefficient : the sparsity coefficient for H updates in the NMFs computed.
        normalize : whether or not H matrices are normalized through the NMFs.
        deterministic : boolean indicating if the nmfs are deterministics or not.
    """
    Hs = []
    n_nmf = len(Xs)
    costs = []

    if type(betas) == int:
        betas = [betas for _ in range(n_nmf)] 
    if type(update_rules) == str:
        update_rules = [update_rules for _ in range(n_nmf)] 
    if type(n_iters) == int:
        n_iters = [n_iters for _ in range(n_nmf)] 

    t = time.time()
    for i in tqdm(range(n_nmf)):
        X = Xs[i]

        if Hs_init is None:
            H_0 = tl.random.random_tensor((rank, len(X[0,:]))) + 1e-12
        else :
            H_0 = Hs_init[i]

        if return_costs:
            _, H, c, _ = nnfacnmf.nmf(X, rank, init="custom", U_0=W_casc, V_0=H_0, n_iter_max=n_iters[i], n_stepV=n_stepH, update_rule=update_rules[i], beta=betas[i],
                            sparsity_coefficients=[0.0, sparsity_coefficient], fixed_modes=[1,0], normalize=[False, normalize], verbose=verbose, return_costs=True, deterministic=deterministic)
            Hs.append(H)
            costs.append(c)
        else :
            _, H = nnfacnmf.nmf(X, rank, init="custom", U_0=W_casc, V_0=H_0, n_iter_max=n_iters[i], n_stepV=n_stepH, update_rule=update_rules[i], beta=betas[i],
                                        sparsity_coefficients=[0.0, sparsity_coefficient], fixed_modes=[1,0], normalize=[False, normalize], verbose=verbose, return_costs=False, deterministic=deterministic)
            Hs.append(H)

    t = time.time() - t

    if return_costs:
        return Hs, t, costs
    
    return Hs, t