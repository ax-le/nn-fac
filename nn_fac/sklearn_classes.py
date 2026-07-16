import numpy as np
import pathlib
import copy

import nn_fac.nmf as nmf
import nn_fac.deep_nmf as dnmf
import nn_fac.multilayer_nmf as mlnmf

def to_str(x):
    if isinstance(x, str):
        return x
    elif isinstance(x, int) or isinstance(x, float):
        return str(x)
    elif isinstance(x, list) or isinstance(x, np.ndarray):
        return "_".join([to_str(a) for a in x])
    else:
        raise ValueError(f"Type not supported: {type(x)}")

class BaseNMF():
    def __init__(self, rank, cache_path = None, init="nndsvd", update_rule = "hals", beta = 2, tol = 1e-8, max_iter = 200, verbose = False):
        self.rank = copy.copy(rank)
        self.cache_path = copy.copy(cache_path)
        self.init = copy.copy(init)
        self.update_rule = copy.copy(update_rule)
        self.beta = copy.copy(beta)
        self.tol = copy.copy(tol)
        self.max_iter = copy.copy(max_iter)
        self.verbose = copy.copy(verbose)

        #self.generic_save_name = f"init_{self.init}_update_rule_{self.update_rule}_beta_{self.beta}_max_iter_{self.max_iter}"
    
    def save(self, data_idx, W, H, errors, toc):
        pathlib.Path(self.dir_save_path).mkdir(parents=True, exist_ok=True)
        np.savez(f"{self.dir_save_path}/{data_idx}_{self.generic_save_name}", W=W, H=H, errors=errors, toc=toc)

    def load(self, data_idx, return_errors = False):
        dict_loaded = np.load(f"{self.dir_save_path}/{data_idx}_{self.generic_save_name}.npz")#, allow_pickle = True)
        W = dict_loaded['W']
        H = dict_loaded['H']
        if return_errors:
            return W, H, dict_loaded['errors'], dict_loaded['toc']
        else:
            return W, H

    def compute(self, data, data_idx = None, return_errors = False):
        if self.cache_path is not None:
            assert data_idx is not None, "data_idx must be provided when using cache."
            try:
                content = self.load(data_idx, return_errors = return_errors)
                if return_errors:
                    W, H, errors, toc = content
                else:
                    W, H = content
            except FileNotFoundError:
                W, H, errors, toc = self._compute_this_method(data)
                self.save(data_idx, W, H, errors, toc)
        else:
            W, H, errors, toc = self._compute_this_method(data)
        
        if return_errors:
            return W, H, errors, toc
        else:
            return W, H
                
    def _compute_this_method(self, data):
        raise NotImplementedError("To be implemented in child classes.")

    def select_two_W_in_multinmf(self, W_list):
        # Return lower and upper matrices
        if type(W_list) is np.ndarray:
            return W_list, W_list
        elif type(W_list) is list:
            if len(W_list) == 2:
                return W_list[0], W_list[1]
            else:
                return W_list[0], W_list[-1]
        else:
            raise ValueError(f"W_list is of an unknown type: {type(W_list)}")
    
class NMF(BaseNMF):
    def __init__(self, rank, cache_path = None, init="nndsvd", update_rule = "mu", beta = 1, tol = 1e-6, max_iter = 200, verbose = False):
        super().__init__(rank, cache_path, init, update_rule, beta, tol, max_iter, verbose)

        self.method_name = "nmf"
        self.dir_save_path = f"{self.cache_path}/{self.method_name}/{to_str(self.rank)}"

        self.generic_save_name = f"init_{self.init}_updaterule_{self.update_rule}_beta_{self.beta}_{self.max_iter}"

        self.diplay_name = f"{self.method_name}_{self.rank}_{self.init}_{self.update_rule}_{self.beta}_maxiter{self.max_iter}"

    def _compute_this_method(self, data):
        rank = copy.copy(self.rank)
        init = copy.copy(self.init)
        n_iter_max = copy.copy(self.max_iter)
        tol = copy.copy(self.tol)
        update_rule = copy.copy(self.update_rule)
        beta = copy.copy(self.beta)

        W, H, errors, toc = nmf.nmf(data, rank = rank, init = init, n_iter_max = n_iter_max, tol = tol, 
                                    update_rule = update_rule, beta = beta, normalize = [False, True], deterministic = True,
                                    verbose = self.verbose,return_costs=True) # Always set return_errors to True for persisting the errors.
        return W, H, errors, toc

class BaseMultiNMF(BaseNMF):
    def __init__(self, rank, cache_path = None, init="nndsvd", update_rule = "hals", beta = 2, tol = 1e-8, max_iter = 200, verbose = False):
        super().__init__(rank, cache_path, init, update_rule, beta, tol, max_iter, verbose)

        #self.generic_save_name = f"init_{self.init}_update_rule_{self.update_rule}_beta_{self.beta}_max_iter_{self.max_iter}"
    
    def save(self, data_idx, W, H, errors, toc):
        pathlib.Path(self.dir_save_path).mkdir(parents=True, exist_ok=True)
        np.savez(f"{self.dir_save_path}/W_{data_idx}_{self.generic_save_name}", *W)
        np.savez(f"{self.dir_save_path}/H_{data_idx}_{self.generic_save_name}", *H)
        np.savez(f"{self.dir_save_path}/errors_{data_idx}_{self.generic_save_name}", errors=errors, toc=toc)

    def load(self, data_idx, return_errors = False):
        W_file = np.load(f"{self.dir_save_path}/W_{data_idx}_{self.generic_save_name}.npz")#, allow_pickle = True)
        W = []
        for arr in W_file.files:
            W.append(W_file[arr])

        H_file = np.load(f"{self.dir_save_path}/H_{data_idx}_{self.generic_save_name}.npz")#, allow_pickle = True)
        H = []
        for arr in H_file.files:
            H.append(H_file[arr])
        
        if return_errors:
            data_errors = np.load(f"{self.dir_save_path}/errors_{data_idx}_{self.generic_save_name}.npz")#, allow_pickle = True)
            return W, H, data_errors['errors'], data_errors['toc']
        else:
            return W, H
    
class DeepNMF(BaseMultiNMF):
    def __init__(self, rank, cache_path = None, init="multilayer_nmf", init_multilayer="nndsvd", update_rule = "mu", beta = 1, tol = 1e-6, max_iter = 200, max_iter_init_mlnmf = 200, verbose = False):
        assert beta == 1, "DeepNMF only supports beta=1 (for now)."
        assert update_rule == "mu", "DeepNMF only supports Multiplicative Updates (for now)."
        super().__init__(rank, cache_path, init, update_rule, beta, tol, max_iter, verbose)

        self.method_name = "deep_nmf"
        self.dir_save_path = f"{self.cache_path}/{self.method_name}/{to_str(self.rank)}"

        self.max_iter_init_mlnmf = copy.copy(max_iter_init_mlnmf)
        self.init = copy.copy(init)
        self.init_multilayer = copy.copy(init_multilayer)

        if self.init == "multilayer_nmf":
            str_init = f"{self.init}_{self.init_multilayer}"
            str_maxiter = f"maxiter_deep_{self.max_iter}_maxiter_multilayer_{self.max_iter_init_mlnmf}"
        else:
            str_init = init
            str_maxiter = f"maxiter_deep_{self.max_iter}"

        self.generic_save_name = f"init_{str_init}_updaterule_{self.update_rule}_beta_{self.beta}_{str_maxiter}"

        self.diplay_name = f"{self.method_name}_{to_str(self.rank)}_{str_init}_{self.update_rule}_{self.beta}_{str_maxiter}"
   
    def _compute_this_method(self, data):
        ranks = copy.copy(self.rank)
        n_iter_max_each_nmf = copy.copy(self.max_iter_init_mlnmf)
        n_iter_max_deep_loop = copy.copy(self.max_iter)
        init = copy.copy(self.init)
        init_multi_layer = copy.copy(self.init_multilayer)

        W, H, erorrs, toc = dnmf.deep_KL_NMF(data, all_ranks = ranks, n_iter_max_each_nmf = n_iter_max_each_nmf, n_iter_max_deep_loop = n_iter_max_deep_loop,
                                             init = init, init_multi_layer = init_multi_layer, verbose = self.verbose, return_errors=True) # Always set return_errors to True for persisting the errors.
        return W, H, erorrs, toc 
        
class MultilayerNMF(BaseMultiNMF):
    def __init__(self, rank, cache_path = None, init="nndsvd", update_rule = "mu", beta = 1, tol = 1e-6, max_iter = 200, verbose = False):
        assert update_rule == "mu", "Multilayer only supports Multiplicative Updates (for now)."
        super().__init__(rank, cache_path, init, update_rule, beta, tol, max_iter, verbose)

        self.method_name = "multilayer_nmf"
        self.dir_save_path = f"{self.cache_path}/{self.method_name}/{to_str(self.rank)}"

        self.generic_save_name = f"init_{self.init}_updaterule_{self.update_rule}_beta_{self.beta}_maxiter{self.max_iter}"
        self.diplay_name = f"{self.method_name}_{to_str(self.rank)}_{init}_{self.update_rule}_{self.beta}_{max_iter}"

    def _compute_this_method(self, data):
        ranks = copy.copy(self.rank)
        beta = copy.copy(self.beta)
        n_iter_max_each_nmf = copy.copy(self.max_iter)
        init_each_nmf = copy.copy(self.init)

        W, H, erorrs, toc = mlnmf.multilayer_beta_NMF(data, all_ranks = ranks, beta = beta, n_iter_max_each_nmf = n_iter_max_each_nmf, init_each_nmf = init_each_nmf, verbose = self.verbose, return_errors=True) # Always set return_errors to True for persisting the errors.
        return W, H, erorrs, toc