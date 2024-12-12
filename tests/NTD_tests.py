# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:47:27 2020

@author: amarmore
"""
# Tests on development of the NTD algorithm.

import unittest
import tensorly as tl
import random
import numpy as np
from nn_fac.ntd import ntd, ntd_mu
import nn_fac.utils.errors as err

class NTDTests(unittest.TestCase):
    
    def setUp(self):
        """
        setUp function, not to redefine the objects in each test function.
        """
        np.random.seed(0)
        random.seed(0)
       
        self.random_ranks = (random.randint(3,10), random.randint(3,10), random.randint(3,10))
        self.random_shape_tens = (random.randint(20,100), random.randint(20,100), random.randint(20,100))

        self.factors_0 = np.random.rand(self.random_shape_tens[0], self.random_ranks[0])
        self.factors_1 = np.random.rand(self.random_shape_tens[1], self.random_ranks[1])
        self.factors_2 = np.random.rand(self.random_shape_tens[2], self.random_ranks[2])
        self.core = np.random.rand(self.random_ranks[0],self.random_ranks[1],self.random_ranks[2])
        self.init_by_product_tensor = tl.tenalg.multi_mode_dot(self.core, [self.factors_0, self.factors_1, self.factors_2])

        self.random_tucker = tl.abs(tl.random.random_tucker(self.random_shape_tens, self.random_ranks, full=True, random_state=0)) + 1e-2 * np.random.rand(self.random_shape_tens[0], self.random_shape_tens[1], self.random_shape_tens[2])
    
    # %% Normal computation
    def test_invalid_ranks_values(self):
        with self.assertRaises(err.InvalidRanksException):
            core, facs, errs, toc = ntd(self.init_by_product_tensor, [3,4], init = "random", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])

    def test_invalid_init_values(self):
        with self.assertRaises(err.InvalidInitializationType):
            core, facs, errs, toc = ntd(self.init_by_product_tensor, [2,4,5], init = "string", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])

    def test_invalid_custom_init_values(self):
        with self.assertRaises(err.CustomNotEngouhFactors):
            core, facs, errs, toc = ntd(self.init_by_product_tensor, [2,4,5], init = "custom", factors_0 = [self.factors_0, self.factors_1], return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])
        
        with self.assertRaises(err.CustomNotValidFactors):
            core, facs, errs, toc = ntd(self.init_by_product_tensor, [2,4,5], init = "custom", factors_0 = [self.factors_0, self.factors_1, None], return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])
            
        with self.assertRaises(err.CustomNotValidCore):
            core, facs, errs, toc = ntd(self.init_by_product_tensor, [2,4,5], init = "custom", factors_0 = [self.factors_0,self.factors_1,self.factors_2], core_0 = None, return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None, None], normalize = [False, False, False, False])
    
    # %% Check that update rules indeed decrease the loss
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
    def preserve_error_n_iterations(self, tensor, ranks, nb_iter = 2, n_iter_ntd = 100, init = "random",
                                    sparsity_coefficients = [None, None, None, None], 
                                    normalize = [False,False,False,False], deterministic = True):
        
        first_iteration = ntd(tensor, ranks, init = init, n_iter_max=n_iter_ntd, tol=1e-6,
                    sparsity_coefficients = sparsity_coefficients, fixed_modes = [], normalize = normalize,
                    verbose=False, return_costs=True, deterministic = deterministic)
        for i in range(nb_iter - 1):
            this_try = ntd(tensor, ranks, init = init, n_iter_max=n_iter_ntd, tol=1e-6,
                    sparsity_coefficients = sparsity_coefficients, fixed_modes = [], normalize = normalize,
                    verbose=False, return_costs=True, deterministic = deterministic)
            if (first_iteration[2][-1] - this_try[2][-1]) != 0:
                return False
        return True
    
    def test_light_determinism(self):
        """
        Verifies that the "determinist" argument is correct, meaning that the result is deterministic.
        This is the "Light" test, relatively fast to compute, but less complete than "heavy testing" function.
        """
        self.assertTrue(self.preserve_error_n_iterations(self.random_tucker, [5,5,5], nb_iter = 2,
                                                  init = "random", deterministic = True))
        self.assertTrue(self.preserve_error_n_iterations(self.random_tucker, [5,5,5], nb_iter = 2, 
                                                init = "random", deterministic = True))
        self.assertTrue(self.preserve_error_n_iterations(self.random_tucker, [5,5,5], nb_iter = 2, 
                                                init = "tucker", deterministic = True))
        self.assertTrue(self.preserve_error_n_iterations(self.random_tucker, [5,5,5], nb_iter = 2, 
                                                init = "tucker", deterministic = True))
        self.assertFalse(self.preserve_error_n_iterations(self.random_tucker, [5,5,5], nb_iter = 2, 
                                                init = "random", deterministic = False))
        self.assertFalse(self.preserve_error_n_iterations(self.random_tucker, [5,5,5], nb_iter = 2, 
                                                init = "random", deterministic = False))
    
    # # # Heavy testing, long to compute, prefer it to be commented when running the tests.
    # # def test_heavy_determinism(self):
    # #     """
    # #     Verifies that the "determinist" argument is correct, meaning that the result is deterministic.
    # #     This is the "Heavy" test, robuster and harder to verify (much confident that the result is correct",
    # #     but long to compute.
    # #     """
    # #     self.assertTrue(self.preserve_error_n_iterations(self.random_tucker, [17,18,19], nb_iter = 5,
    # #                                               init = "random", deterministic = True))
    # #     self.assertTrue(self.preserve_error_n_iterations(self.random_tucker, [17,18,19], nb_iter = 5, 
    # #                                             init = "random", deterministic = True))
    # #     self.assertTrue(self.preserve_error_n_iterations(self.random_tucker, [17,18,19], nb_iter = 5, 
    # #                                             init = "tucker", deterministic = True))
    # #     self.assertTrue(self.preserve_error_n_iterations(self.random_tucker, [17,18,19], nb_iter = 5, 
    # #                                             init = "tucker", deterministic = True))
    # #     self.assertFalse(self.preserve_error_n_iterations(self.random_tucker, [17,18,19], nb_iter = 5, 
    # #                                             init = "random", deterministic = False))
    # #     self.assertFalse(self.preserve_error_n_iterations(self.random_tucker, [17,18,19], nb_iter = 5, 
    # #                                             init = "random", deterministic = False))

    # %% Test of NTD on one particular test
    def test_decomposition_hals(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.random_tucker[0][0][0], 21.974433828159626)

        core, factors, cost_fct_vals, toc = ntd(self.random_tucker, self.random_ranks, init = "random", n_iter_max = 10, tol = 1e-8,
                                                    deterministic = True,sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                    verbose = False, return_costs = True)
        
        # Checking factors
        self.assertAlmostEqual(factors[0][0][0], 0.15008560444598218)
        self.assertAlmostEqual(factors[1][0][0], 0.5932534337969397)
        self.assertAlmostEqual(factors[2][0][0], 0.5827772361357559)
        self.assertAlmostEqual(core[0,0,0], 0.644143536068335)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 0.0009901099693517598)
        self.assertAlmostEqual(cost_fct_vals[-1], 0.00012342263436269062)

    def test_decomposition_mu_beta2(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.random_tucker[0][0][0], 21.974433828159626)

        core, factors, cost_fct_vals, toc = ntd_mu(self.random_tucker, self.random_ranks, init = "random", n_iter_max = 10, tol = 1e-8, beta = 2,
                                                    deterministic = True,sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                    verbose = False, return_costs = True)
        
        # Checking factors
        self.assertAlmostEqual(factors[0][0][0], 0.4918486535796236)
        self.assertAlmostEqual(factors[1][0][0], 0.7948442026225503)
        self.assertAlmostEqual(factors[2][0][0], 0.5126011046374159)
        self.assertAlmostEqual(core[0,0,0], 0.6401491933742705)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 45250.45478060806)
        self.assertAlmostEqual(cost_fct_vals[-1], 42246.166217618826)

    def test_decomposition_mu_beta1(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.random_tucker[0][0][0], 21.974433828159626)

        core, factors, cost_fct_vals, toc = ntd_mu(self.random_tucker, self.random_ranks, init = "random", n_iter_max = 10, tol = 1e-8, beta = 1,
                                                    deterministic = True,sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                    verbose = False, return_costs = True)
        
        # Checking factors
        self.assertAlmostEqual(factors[0][0][0], 0.5590567522139028)
        self.assertAlmostEqual(factors[1][0][0], 0.7579973274629266)
        self.assertAlmostEqual(factors[2][0][0], 0.4717461273286412)
        self.assertAlmostEqual(core[0,0,0], 0.6397194128224372)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 2636.709914183355)
        self.assertAlmostEqual(cost_fct_vals[-1], 2474.7683618573797)

    def test_decomposition_mu_beta0(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.random_tucker[0][0][0], 21.974433828159626)

        core, factors, cost_fct_vals, toc = ntd_mu(self.random_tucker, self.random_ranks, init = "random", n_iter_max = 10, tol = 1e-8, beta = 0,
                                                    deterministic = True,sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                    verbose = False, return_costs = True)
        
        # Checking factors
        self.assertAlmostEqual(factors[0][0][0], 0.6112093668857077)
        self.assertAlmostEqual(factors[1][0][0], 0.7871188331570661)
        self.assertAlmostEqual(factors[2][0][0], 0.43120029540647997)
        self.assertAlmostEqual(core[0,0,0], 0.6350545357176396)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 21192.73157719653)
        self.assertAlmostEqual(cost_fct_vals[-1], 189.0160787574748)



# %% Run tests
if __name__ == '__main__':
    unittest.main()