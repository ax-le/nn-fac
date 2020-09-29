# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:47:27 2020

@author: amarmore
"""
# Tests on development of the NTD algorithm.

import context

import unittest
import tensorly as tl
import random
import numpy as np
from NTD import ntd
import errors as err

class NTDTests(unittest.TestCase):
    
    def setUp(self):
        """
        setUp function, not to redefine the objects in each test function.
        """
        self.tensor = tl.tensor(np.random.rand(random.randint(20,100), random.randint(20,100), random.randint(20,100)))
        
        self.random_ranks = (random.randint(20,40), random.randint(20,40), random.randint(20,40))
        self.random_shape_tens = (random.randint(50,100), random.randint(50,100), random.randint(50,100))
        self.factors_0 = random.randint(1,10) * np.random.rand(self.random_shape_tens[0], self.random_ranks[0])
        self.factors_1 = random.randint(1,10) * np.random.rand(self.random_shape_tens[1], self.random_ranks[1])
        self.factors_2 = random.randint(1,10) * np.random.rand(self.random_shape_tens[2], self.random_ranks[2])
        self.core = np.random.rand(self.random_ranks[0],self.random_ranks[1],self.random_ranks[2])
        self.init_by_product_tensor = tl.tenalg.multi_mode_dot(self.core, [self.factors_0, self.factors_1, self.factors_2])
    
    # %% Normal computation
    def test_invalid_ranks_values(self):
        with self.assertRaises(err.InvalidRanksException):
            core, facs, errs, toc = ntd(self.init_by_product_tensor, [30,10], init = "random", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])
        with self.assertRaises(err.InvalidRanksException):
            core, facs, errs, toc = ntd(self.init_by_product_tensor, [30,120,40], init = "random", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])
    
    def test_invalid_init_values(self):
        with self.assertRaises(err.InvalidInitializationType):
            core, facs, errs, toc = ntd(self.init_by_product_tensor, [30,10,10], init = "string", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])

    def test_invalid_custom_init_values(self):
        with self.assertRaises(err.CustomNotEngouhFactors):
            core, facs, errs, toc = ntd(self.init_by_product_tensor, [30,10,10], init = "custom", factors_0 = [self.factors_0, self.factors_1], return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])
        
        with self.assertRaises(err.CustomNotValidFactors):
            core, facs, errs, toc = ntd(self.init_by_product_tensor, [30,10,10], init = "custom", factors_0 = [self.factors_0, self.factors_1, None], return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])
            
        with self.assertRaises(err.CustomNotValidCore):
            core, facs, errs, toc = ntd(self.init_by_product_tensor, [30,10,10], init = "custom", factors_0 = [self.factors_0,self.factors_1,self.factors_2], core_0 = None, return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])
    
    # %% Good decomposition
    def strictly_decreasing(self, L):
        return all(x>y for x, y in zip(L, L[1:]))
    
    def test_good_random_decomp(self):
        core, facs, errs, toc = ntd(self.init_by_product_tensor, self.random_ranks, init = "random", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])
        self.assertAlmostEqual(errs[-1], 0, places = 2)
        self.assertTrue(self.strictly_decreasing(errs))
        
    def test_good_tucker_decomp(self):
        core, facs, errs, toc = ntd(self.init_by_product_tensor, self.random_ranks, init = "tucker", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])
        self.assertAlmostEqual(errs[-1], 0, places = 2)
        self.assertTrue(self.strictly_decreasing(errs))
    
    def test_good_custom_decomp(self):
        core, facs, errs, toc = ntd(self.init_by_product_tensor + np.random.random(self.random_shape_tens), self.random_ranks, init = "custom", factors_0 = [self.factors_0, self.factors_1, self.factors_2], core_0 = self.core,
                                    return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])
        self.assertAlmostEqual(errs[-1], 0, places = 2)
        self.assertTrue(self.strictly_decreasing(errs))
                
    # %% Determinism testing
    def preserve_error_n_iterations(self, tensor, ranks, nb_iter = 2, n_iter_ntd = 100, hals = False, init = "random",
                                    sparsity_coefficients = [None, None, None, None], 
                                    normalize = [False,False,False,False], deterministic = True):
        
        first_iteration = ntd(tensor, ranks, init = init, n_iter_max=n_iter_ntd, tol=1e-6,
                    sparsity_coefficients = sparsity_coefficients, fixed_modes = [], normalize = normalize, hals = hals,
                    verbose=False, return_costs=True, deterministic = deterministic)
        for i in range(nb_iter - 1):
            this_try = ntd(tensor, ranks, init = init, n_iter_max=n_iter_ntd, tol=1e-6,
                    sparsity_coefficients = sparsity_coefficients, fixed_modes = [], normalize = normalize, hals = hals,
                    verbose=False, return_costs=True, deterministic = deterministic)
            if (first_iteration[2][-1] - this_try[2][-1]) != 0:
                return False
        return True
    
    def test_light_determinism(self):
        """
        Verifies that the "determinist" argument is correct, meaning that the result is deterministic.
        This is the "Light" test, relatively fast to compute, but less complete than "heavy testing" function.
        """
        self.assertTrue(self.preserve_error_n_iterations(self.tensor, [5,5,5], nb_iter = 2,
                                                  init = "random", deterministic = True, hals = False))
        self.assertTrue(self.preserve_error_n_iterations(self.tensor, [5,5,5], nb_iter = 2, 
                                                init = "random", deterministic = True, hals = True))
        self.assertTrue(self.preserve_error_n_iterations(self.tensor, [5,5,5], nb_iter = 2, 
                                                init = "tucker", deterministic = True, hals = False))
        self.assertTrue(self.preserve_error_n_iterations(self.tensor, [5,5,5], nb_iter = 2, 
                                                init = "tucker", deterministic = True, hals = True))
        self.assertFalse(self.preserve_error_n_iterations(self.tensor, [5,5,5], nb_iter = 2, 
                                                init = "random", deterministic = False, hals = False))
        self.assertFalse(self.preserve_error_n_iterations(self.tensor, [5,5,5], nb_iter = 2, 
                                                init = "random", deterministic = False, hals = True))
    
    # # Heavy testing, long to compute, prefere it to be commented when running the tests.
    # def test_heavy_determinism(self):
    #     """
    #     Verifies that the "determinist" argument is correct, meaning that the result is deterministic.
    #     This is the "Heavy" test, robuster and harder to verify (much confident that the result is correct",
    #     but long to compute.
    #     """
    #     self.assertTrue(self.preserve_error_n_iterations(self.tensor, [17,18,19], nb_iter = 5,
    #                                               init = "random", deterministic = True, hals = False))
    #     self.assertTrue(self.preserve_error_n_iterations(self.tensor, [17,18,19], nb_iter = 5, 
    #                                             init = "random", deterministic = True, hals = True))
    #     self.assertTrue(self.preserve_error_n_iterations(self.tensor, [17,18,19], nb_iter = 5, 
    #                                             init = "tucker", deterministic = True, hals = False))
    #     self.assertTrue(self.preserve_error_n_iterations(self.tensor, [17,18,19], nb_iter = 5, 
    #                                             init = "tucker", deterministic = True, hals = True))
    #     self.assertFalse(self.preserve_error_n_iterations(self.tensor, [17,18,19], nb_iter = 5, 
    #                                             init = "random", deterministic = False, hals = False))
    #     self.assertFalse(self.preserve_error_n_iterations(self.tensor, [17,18,19], nb_iter = 5, 
    #                                             init = "random", deterministic = False, hals = True))

# %% Run tests
if __name__ == '__main__':
    unittest.main()