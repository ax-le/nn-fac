# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:12:33 2019

@author: amarmore
"""

import numpy as np
import time
import nnls
import tensorly as tl
from nimfa.methods import seeding

def parafac_2(tensor_slices, rank, init_with_P, init = "random", W_list_in = None, H = None, D_list_in = None, W_star = None, P_list = None, n_iter_max=100, tol=1e-6,
              sparsity_coefficient = None, fixed_modes = [], normalize = [False, False, False, False],
              verbose=False, return_errors=False):
    
    """
    ========
    PARAFAC2
    ========
    
    Factorization of a third-order tensor T in nonnegative matrices,
    corresponding to a PARAFAC2 decomposition of this tensor. [1]
    
    Denoting T_{(k)} the k-th slice of T, it will be decomposed in three matrices W_k, H and D_k.
    D_k is a diagonal matrix, corresponding to the k-th column of the third matrix of a PARAFAC decomposition.
    All of these matrcies are stored in a list, respectively W_list and D_list, for every slice of the tensor.
    
    More precisely, this algorithm use a penalty constraint to couple the matrices W_k to the matrix W_star [2].
    This facilitates the computation, as the constraint is relaxed.
    
    The factorization problem is solved by fixing alternatively 
    all factors except one, and resolving the problem on this one.
    Factorwise, the problem is reduced to a Nonnegative Least Squares problem.
    
    The optimization problem is, with mathematical notations:
        
        argmin{W_k, W^*, H, D_k, P_k} \sum\limits_{k = 1}^{K} ||T_{(k)} - W_k D_k H||_Fro^2
        + mu_k * (||W_k - P_k W^*||_Fro^2)
        
    and, written with the variables names: 
    
        argmin{W_list, W_star, H, D_list, P_list} \sum\limits_{k = 1}^{K} ||tensor_slices[k] - W_list[k] D_list[k] H||_Fro^2
        + mu_list[k] * (||W_list[k] - P_list[k] W_star||_Fro^2)
        
    With:
        ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
        ||a||_1 = \sum_{i} abs(a_{i}) (Elementwise L1 norm)
    
    More precisely, the chosen optimization algorithm is the HALS [3],
    which updates each factor columnwise, fixing every other columns.

    The order of the factors for "normalize" and "fixed_modes" is as follows:
        0 -> W_list, 1 -> H, 2 -> D_list, 3 -> W_star, 4 -> P_list
    
    Parameters
    ----------
    tensor_slices: list
        list of slices of a nonnegative tensor
    rank: integer
        The rank of the decomposition
    init_with_P: boolean
        Define whereas the PARAFAC2 decomposition must be performed by initializing P_k or W^*:
            True: initialize with the P_k
            False: initialize with W^*
    init: "random" | "nndsvd" | "custom" |
        - If set to random: 
            Initialize with random factors of the correct size.
            The randomization is the uniform distribution in [0,1),
            which is the default from numpy random.
        - If set to nnsvd:
            Corresponds to a Nonnegative Double Singular Value Decomposition
            (NNDSVD) initialization, which is a data based initialization,
            designed for NMF resolution. See [4] for details.
            This nndsvd if performed via the nimfa toolbox [5].
        - If set to custom:
            W_list_in, H, D_list_in, and W_star or P_list, depending on init_with_P,
            will be used for initializing the decomposition
        Default: random        
    W_list_in: list or None
        A custom initialization for the W_k, used only in "custom" init mode.
        Default: None
    H: array or None
        A custom initialization for H, used only in "custom" init mode.
        Default: None
    D_list_in: list or None
        A custom initialization for the D_k, used only in "custom" init mode.
        Default: None
    W_star:
        A custom initialization for the W^*, used only in "custom" init mode and if init_with_P is set to False.
        Default: None
    P_list: list or None
        A custom initialization for the P_k, used only in "custom" init mode  and if init_with_P is set to True.
        Default: None
    n_iter_max: integer
        The maximal number of iteration before stopping the algorithm
        Default: 100
    tol: float
        Threshold on the improvement in reconstruction error.
        Between two iterations, if the reconstruction error difference is 
        below this threshold, the algorithm stops.
        Default: 1e-8
    sparsity_coefficients: float
        The sparsity coefficient on H.
        If set to None, the algorithm is computed without sparsity
        Default: None
    fixed_modes: array of integers (between 0 and 5)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: []
    normalize: array of boolean (5)
        A boolean whereas the factors need to be normalized.
        The normalization is a l_2 normalization on each of the rank components
        Default: []
    verbose: boolean
        Indicates whether the algorithm prints the successive 
        reconstruction errors or not
        Default: False
    return_errors: boolean
        Indicates whether the algorithm should return all reconstruction errors
        and computation time of each iteration or not
        Default: False
        
    Returns
    -------
    U, V: numpy arrays
        Factors of the NMF 
    errors: list
        A list of reconstruction errors at each iteration of the algorithm.
    toc: list
        A list with accumulated time at each iterations
        
    Example
    -------
    np.array(factors): numpy array
        An array containing all the factors computed with PARAFAC decomposition
    errors: list
        A list of reconstruction errors at each iteration of the algorithm.
    toc: list
        A list with accumulated time at each iterations
    
    References
    ----------
    [1]: R. A Harshman. “PARAFAC2: Mathematical and technical notes”,
    UCLA working papers in phonetics 22.3044, 1972.
    
    [2]: J. E. Cohen and R. Bro, Nonnegative PARAFAC2: A Flexible Coupling Approach,
    DOI: 10.1007/978-3-319-93764-9_9
    
    [3]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.
    
    [4]: Christos Boutsidis and Efstratios Gallopoulos. "SVD based
    initialization: A head start for nonnegative matrix factorization",
    Pattern Recognition 41.4 (2008), pp. 1350{1362.
    
    [5]: Blalz Zupan et al. "Nimfa: A python library for nonnegative matrix
    factorization", Journal of Machine Learning Research 13.Mar (2012),
    pp. 849{853.
    
    """
    
    nb_channel = len(tensor_slices)
    r, n = tensor_slices[0].shape
    
    W_list = []
    D_list = []
    
    if init.lower() == "random":
        H = np.random.rand(rank, n)
        for k in range(nb_channel):
            W_list.append(np.random.rand(r, rank))
            D_list.append(np.diag(np.random.rand(rank)))
        D_list = np.array(D_list)
        if init_with_P:
            P_list = [np.random.rand(r, rank) for i in range(nb_channel)]
        else:
            W_star = np.random.rand(r, rank)

        
    elif init.lower() == "nndsvd":
        for k in range(nb_channel):
            W_k, H = seeding.Nndsvd().initialize(tensor_slices[k], rank, {'flag': 0})
            W_list.append(W_k)
            D_list.append(np.diag(np.random.rand(rank)))
        D_list = np.array(D_list)
        if init_with_P:
            zero_padded_identity = np.identity(r)
            zero_padded_identity = zero_padded_identity[:,0:rank]
            P_list = [zero_padded_identity for i in range(nb_channel)]
        else:
            W_star_local = np.zeros(W_list[0].shape)
            for k in range(nb_channel):
                W_star_local += W_list[k]
            W_star = np.divide(W_star_local, k)



    elif init.lower() == "custom":
        if W_list_in is None or H is None or D_list_in is None: # No verification on P_list and W_star as it occurs below
            raise Exception("Custom initialization, but one factor is set to 'None'")
        else:
            W_list = W_list_in.copy()
            D_list = D_list_in.copy()
    
    else:
        raise Exception('Initialization type not understood')

    return compute_parafac_2(tensor_slices, rank, W_list_in = W_list, H_0 = H, D_list_in = D_list, init_with_P = init_with_P, W_star_in = W_star, P_list_in = P_list, n_iter_max=n_iter_max, tol=tol,
              sparsity_coefficient = sparsity_coefficient, fixed_modes = fixed_modes, normalize = [False, False, False, False],
              verbose=verbose, return_errors=return_errors)


# Author : Jeremy Cohen, modified by Axel Marmoret
def compute_parafac_2(tensor_slices, rank, W_list_in, H_0, D_list_in, init_with_P, W_star_in = None, P_list_in = None, n_iter_max=100, tol=1e-8,
              sparsity_coefficient = None, fixed_modes = [], normalize = [False, False, False, False],
              verbose=False, return_errors=False):
    ''' Perform PARAFAC 2 as a flexible coupling approach,
     See J. E. Cohen and R. Bro, Nonnegative PARAFAC2: A Flexible Coupling Approach,
     DOI: 10.1007/978-3-319-93764-9_9

    Commentaires Jérémy:
    - Important: augmenter mu à l'itération 1. La valeur initiale est vonlontairement faible.
    - Tu peux calculer l'erreur de couplage \|W_k - P_k W_star\| / \|W_k\| afin de
      savoir si tu dois jouer avec les paramètres de couplage.
    - donner l'option d'initialer les Dk
    - donner une valeur max pour mu
    - Tu peux calculer l'erreur initiale en utilisant les calculs ligne 332
     '''
     
    """
    # TODO : doc : bien préciser ce que sont les modes dans normalize (W, H, D, W^*, P_k), et dans fixed_modes
    # TODO : Gérer coefficient de parcimonie, différent des autres fonctions pour l'instant
    # TODO : Bien expliquer init_with_p
    """
    
    # initialization - store the input varaibles    
    nb_channel = len(tensor_slices)
    
    W_list = W_list_in.copy()
    H = H_0.copy()
    D_list = D_list_in.copy()

    if W_star_in is None:
        W_star = None
    else:
        W_star =  W_star_in.copy()
    
    if P_list_in is None:
        P_list= None
    else:
        P_list = P_list_in.copy()
    
    # initialization - declare local varaibles
    rec_errors = []
    tic = time.time()
    toc = []
    norm_slices = []
    couple_error = []
    couple_errors = []
    increasing_mu = True
   
    mu_list = np.zeros(nb_channel)
    
    # Initialization - mu_list
    for k in range(nb_channel):
        mu_list[k] = (np.linalg.norm(tensor_slices[k]-(W_list[k]@D_list[k]@H), ord='fro') ** 2)/(10*np.linalg.norm(W_list[k], ord='fro') ** 2)
        
        # Storing the norm of every slice
        norm_slices.append(np.linalg.norm(tensor_slices[k], ord='fro'))

    for iteration in range(n_iter_max):
        
        if iteration == 0:
            previous_rec_error = None
            previous_couple_error = None
        else:
            previous_rec_error = rec_errors[-1]
            previous_couple_error = couple_errors[-1]
        
        # Increase mu
        if iteration == 1:
            for k in range(nb_channel):
                mu_list[k] = 0.2 * np.linalg.norm(tensor_slices[k]-W_list[k]@D_list[k]@H, ord='fro') / couple_error[k]
        
        if iteration == 2:
            increasing_mu = True
            
        W_list, H, D_list, W_star, P_list, mu_list, rec_error, couple_error, increasing_mu = one_step_parafac2(tensor_slices, rank, W_list, H, D_list, mu_list, norm_slices, 
                                                                                                       previous_rec_error, previous_couple_error, increasing_mu = increasing_mu,
                                                                                                       init_with_P = init_with_P, P_list_in = P_list, W_star_in = W_star,
                                                                                                       sparsity_coefficient = sparsity_coefficient, fixed_modes = fixed_modes, normalize = normalize)

        # Store the computation time
        toc.append(time.time() - tic)
        
        rec_errors.append(rec_error)
        couple_errors.append(couple_error)
        
        if verbose:
            if iteration == 0:
                print('reconstruction error={}'.format(rec_errors[iteration]))
                """for k in range(nb_channel):
                    print('couple_error for channel {} = {}'.format(k, (couple_errors[iteration])[k]))"""
            else:
                if rec_errors[-2] - rec_errors[-1] > 0:
                    print('reconstruction error={}, variation={}.'.format(
                            rec_errors[-1], rec_errors[-2] - rec_errors[-1]))
                else:
                    if rec_errors[-2] - rec_errors[-1] > 0:
                        print('reconstruction error={}, variation={}.'.format(
                                rec_errors[-1], rec_errors[-2] - rec_errors[-1]))
                    else:
                        print('\033[91m' + 'reconstruction error={}, variation={}.'.format(
                                rec_errors[-1], rec_errors[-2] - rec_errors[-1]) + '\033[0m')
    
                """for k in range(nb_channel):
                    if (couple_errors[-2])[k] - (couple_errors[-1])[k] > 0:
                        print('couple_error for channel {} = {}, variation={}.'.format(k, (couple_errors[-1])[k],
                          (couple_errors[-2])[k] - (couple_errors[-1])[k]))
                    else:
                        print('\033[91m' + 'couple_error for channel {} = {}, variation={}.'.format(k, (couple_errors[-1])[k],
                          (couple_errors[-2])[k] - (couple_errors[-1])[k]) + '\033[0m')"""
                
                
        if iteration > 0 and abs(rec_errors[-2] - rec_errors[-1]) < tol:
            # Stop condition: relative error between last two iterations < tol
            if verbose:
                print('converged in {} iterations.'.format(iteration))
            break
            

    if return_errors:
        return W_list, np.array(H), D_list, rec_errors, toc
    else:
        return W_list, np.array(H), D_list
    
def one_step_parafac2(slices, rank, W_list_in, H_in, D_list_in, mu_list_in, norm_slices,
                      previous_rec_error, previous_couple_error, increasing_mu = True,
                      init_with_P = True, P_list_in = None, W_star_in = None,
                      sparsity_coefficient = None, fixed_modes = [], normalize = [False, False, False, False, False]):

    """ One pass of PARAFAC 2 update on all channels

    Update the factors by solving least squares problems per factor.

    Parameters
    ----------
    slices : list of array
        List of the slices of the spectrogram (for each channel)
    rank : int
        Rank of the decomposition.
    W_list_in : list of array
        Current estimates for the PARAFAC 2 decomposition of the factors W_k (for each channel)
    D_list_in : list of diagonal arrays
        Current estimates for the PARAFAC 2 decomposition of the factors D_k (for each channel)
    H_in : Array
        Current estimate for the PARAFAC 2 decomposition of the factor H
    mu_list_in : list of float
        Parameter for the weight of the latent factor
    norm_slices : list of norms
        The Frobenius norms of each slice of the spectrogram (for each channel)
        
    previous_rec_error
    previous_couple_error
    increasing_mu = True,
                      init_with_P = True, 
                      P_list: list of arrays
        Current estimate for the PARAFAC 2 decomposition of the factors P_k
                      P_list_in = None, W_star_in = None,
                      sparsity_coefficient = None, fixed_modes = [], normalize = [False, False, False, False, False]
            
    Returns
    -------
    W_list, D_list, H, W_star, mu_list:
        The updated factors
    rec_error : float
        residual error after the ALS steps.


    """
    W_list = W_list_in.copy()
    D_list = D_list_in.copy()
    H = H_in.copy()
    mu_list = mu_list_in.copy()
    rec_error = 0
    nb_channel = len(W_list)
    
    # Tous les append doivent etre reinit ici
    if P_list_in is None and W_star_in is None:
        raise ValueError('The list of P_k and W^* are both to None: one has to be set for the operation.')
        
    elif init_with_P == True and P_list_in is None:
        raise ValueError('PARAFAC2 is set with the init of P_k, but they are set to None.')
    elif init_with_P == False and W_star_in is None:
        raise ValueError('PARAFAC2 is set with the init of W^*, but it is set to None.')


    # The initialization is made with the P_k
    if init_with_P:
        P_list = P_list_in.copy()
        W_star = compute_W_star(P_list, W_list, mu_list, nb_channel, normalize = True)
        if 4 in fixed_modes:
            P_list = compute_P_k(W_list, W_star, nb_channel)
    
    # The initialization is made with W^*
    else:
        W_star = W_star_in
        P_list = compute_P_k(W_list, W_star, nb_channel)
        if 3 in fixed_modes:
            W_star = compute_W_star(P_list, W_list, mu_list, nb_channel, normalize = normalize[3])

    for k in range(nb_channel):
        if 0 not in fixed_modes:
            # Update W_k
            
            tic = time.time()
                    
            DkH = D_list[k]@H
            
            VVt = np.dot(DkH, np.transpose(DkH))
            VMt = np.dot(DkH, np.transpose(slices[k]))
            
            timer = time.time() - tic
            
            W_list[k] = np.transpose(nnls.hals_nnls_fitting_acc(VMt, VVt, np.transpose(W_list[k]), np.transpose(P_list[k]@W_star), mu_list[k],
                                                                maxiter=100, atime=timer, alpha=0.5, delta=0.01,
                                                                normalize = normalize[0], nonzero = False)[0])

        if 2 not in fixed_modes:
            # Update D_k
            
            tic = time.time()

            khatri = tl.tenalg.khatri_rao([W_list[k], H.T])
                            
            UtU = np.transpose(khatri)@khatri
    
            # flattening line by line, so no inversion of the matrices in the Khatri-Rao product.
            UtM_local = (khatri.T)@(slices[k].flatten())
    
            UtM = np.reshape(UtM_local, (UtM_local.shape[0],1))
    
            timer = time.time() - tic
            
            # Keep only the diagonal coefficients            
            diag_D = np.diag(D_list[k])
            
            # Reshape for having a proper column vector (error in nnls function otherwise)
            diag_D = np.reshape(diag_D, (diag_D.shape[0],1)) # It simply instead becomes a vector column instead of a list
            
            D_list[k] = nnls.hals_nnls_acc(UtM, UtU, diag_D, maxiter=100, atime=timer, alpha=0.5, delta=0.01,
                                          sparsity_coefficient = None, normalize = False, nonzero = False)[0] # All these parameters are not available for a diagonal matrix
            
            # Make the matrix a diagonla one
            D_list[k] = np.diag(np.diag((D_list[k])))
                    
    if normalize[2]:
        for note_index in range(rank):
            norm = np.linalg.norm(D_list[:,note_index], ord='fro')
            if norm == 0:
                D_list[:,note_index, note_index] = [1/(nb_channel ** 2) for k in range(nb_channel)]
            else:
                D_list[:,note_index] /= np.linalg.norm(D_list[:,note_index], ord='fro')

    if 1 not in fixed_modes:
        # Update H

        tic = time.time()
    
        UtU = np.zeros((rank,rank))
        UtM = np.zeros((rank,(slices[0].shape)[1]))
        
        for k in range(nb_channel):
            WkDk = W_list[k]@D_list[k]
            UtU += np.dot(np.transpose(WkDk), WkDk)
            UtM += np.dot(np.transpose(WkDk), slices[k])
        
        timer = time.time() - tic
        
        H = nnls.hals_nnls_acc(UtM, UtU, H, maxiter=100, atime=timer, alpha=0.5, delta=0.01,
                               sparsity_coefficient = sparsity_coefficient, normalize = normalize[1], nonzero = False)[0]    
    
        
    couple_error = []
    
    #if sparsity_coefficient != None:
    #    rec_error = sparsity_coefficient * np.linalg.norm()
    
    for k in range(nb_channel):
        couple_error.append(np.linalg.norm(W_list[k] - P_list[k]@W_star, ord='fro'))
        
        slice_rec_error = np.linalg.norm(slices[k]-W_list[k]@D_list[k]@H)** 2 + (mu_list[k] * couple_error[k]**2) / norm_slices[k]
        rec_error += slice_rec_error
        
        #TODO: Abritrary tolerances, to be changed and passed as constants
        if previous_rec_error != None and previous_couple_error != None:
            if mu_list[k] < 1e6 and (previous_rec_error - rec_error) > 0 and (previous_couple_error[k] - slice_rec_error) > 1e-6 and increasing_mu:
                mu_list[k] *= 1.02
            elif increasing_mu: # Stop increasing mu for the next iterations
                increasing_mu = False

    return W_list, H, D_list, W_star, P_list, mu_list, rec_error, couple_error, increasing_mu


def compute_P_k(W_list, W_star, nb_channel):
    """ A function for updating P_k """
    list_of_P = []
    nb_columns_P = W_star.shape[0]
    for k in range(nb_channel):
        U, S, Vt = np.linalg.svd(W_list[k]@W_star.T)
        list_of_P.append(U[:,0:nb_columns_P]@Vt[0:nb_columns_P,:])
    return list_of_P

def compute_W_star(P_list, W_list, mu_list, nb_channel, normalize = False):
    """ A function for updating W^* """
    nb_columns = W_list[0].shape[1] # Equal to the number of columns of any W
    nb_lines = P_list[0].shape[1] # Equal to the number of columns of any P (or lines of P.T)
    W_star_local_sum = np.zeros((nb_lines, nb_columns))
    for k in range(nb_channel):
        # Store the future computation for W_star
        W_star_local_sum += mu_list[k] * P_list[k].T @ W_list[k]
    # Compute W_star
    local_W_star = W_star_local_sum/np.sum(mu_list)
    if normalize:
        # Normalize W_star columnwise. -> Useless in theory, and causes errors in practice
        for index in range(nb_columns):
            norm = np.linalg.norm(local_W_star[:,index], ord=2)
            if norm != 0:
                local_W_star[:,index] /= norm
    return local_W_star
        