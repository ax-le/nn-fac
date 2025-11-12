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
    def test_decomposition_hals_random_init(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.random_tucker[0][0][0], 21.974433828159626)

        core, factors, cost_fct_vals, toc = ntd(self.random_tucker, self.random_ranks, init = "random", n_iter_max = 10, tol = 1e-8,
                                                    sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                    verbose = False, return_costs = True, deterministic = True, seed=0)
        
        # Checking factors
        self.assertAlmostEqual(factors[0][0][0], 0.5501411956914489)
        self.assertAlmostEqual(factors[1][0][0], 0.9680069293664532)
        self.assertAlmostEqual(factors[2][0][0], 0.965086018254149)
        self.assertAlmostEqual(core[0,0,0], 0.3744157888431357)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 2.6164388105612055e-08)
        self.assertAlmostEqual(cost_fct_vals[-1], 2.603936417799217e-08)

    def test_decomposition_hals_tucker_init(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.random_tucker[0][0][0], 21.974433828159626)

        core, factors, cost_fct_vals, toc = ntd(self.random_tucker, self.random_ranks, init = "tucker", n_iter_max = 10, tol = 1e-8,
                                                update_rule="hals",
                                                    sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                    verbose = False, return_costs = True, deterministic = True, seed=0)
        
         # Checking factors
        self.assertAlmostEqual(factors[0][0][0], 0.16504481330298995)
        self.assertAlmostEqual(factors[1][0][0], 0.09847086272185894)
        self.assertAlmostEqual(factors[2][0][0], 0.11680262111792158)
        self.assertAlmostEqual(core[0,0,0], 11039.862648258559)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 0.00027083233922590056)
        self.assertAlmostEqual(cost_fct_vals[-1], 0.00010638116104305596)

    def test_decomposition_mu_beta2_random_init(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.random_tucker[0][0][0], 21.974433828159626)

        core, factors, cost_fct_vals, toc = ntd(self.random_tucker, self.random_ranks, init = "random", n_iter_max = 10, tol = 1e-8,
                                                update_rule="mu", beta = 2,
                                                    sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                    verbose = False, return_costs = True, deterministic = True, seed=0)
        
        # Checking factors
        self.assertAlmostEqual(factors[0][0][0], 0.5489250094099122)
        self.assertAlmostEqual(factors[1][0][0], 0.9679994929177957)
        self.assertAlmostEqual(factors[2][0][0], 0.9650887516147171)
        self.assertAlmostEqual(core[0,0,0], 0.3744138868288453)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 1.5935015225944391)
        self.assertAlmostEqual(cost_fct_vals[-1], 1.5931775725367523)

    def test_decomposition_mu_beta2_tucker_init(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.random_tucker[0][0][0], 21.974433828159626)

        core, factors, cost_fct_vals, toc = ntd(self.random_tucker, self.random_ranks, init = "tucker", n_iter_max = 10, tol = 1e-8,
                                                update_rule="mu", beta = 2,
                                                    sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                    verbose = False, return_costs = True, deterministic = True, seed=0)
        
        # Checking factors
        self.assertAlmostEqual(factors[0][0][0], 0.1633567459395657)
        self.assertAlmostEqual(factors[1][0][0], 0.09484478066313659)
        self.assertAlmostEqual(factors[2][0][0], 0.1174295516693132)
        self.assertAlmostEqual(core[0,0,0], 11046.430317228587)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 22653.665491321422)
        self.assertAlmostEqual(cost_fct_vals[-1], 21679.048477120345)

    def test_decomposition_mu_beta1_random_init(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.random_tucker[0][0][0], 21.974433828159626)

        core, factors, cost_fct_vals, toc = ntd(self.random_tucker, self.random_ranks, init = "random", n_iter_max = 10, tol = 1e-8,
                                                update_rule="mu", beta = 1,
                                                    sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                    verbose = False, return_costs = True, deterministic = True, seed=0)
        
        # Checking factors
        self.assertAlmostEqual(factors[0][0][0], 0.5489424379755086)
        self.assertAlmostEqual(factors[1][0][0], 0.9679939115774175)
        self.assertAlmostEqual(factors[2][0][0], 0.9650587287572271)
        self.assertAlmostEqual(core[0,0,0], 0.3744133064030978)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 0.12936809612191502)
        self.assertAlmostEqual(cost_fct_vals[-1], 0.1293171172587153)

    def test_decomposition_mu_beta0_random_init(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.random_tucker[0][0][0], 21.974433828159626)

        core, factors, cost_fct_vals, toc = ntd(self.random_tucker, self.random_ranks, init = "random", n_iter_max = 10, tol = 1e-8,
                                                   update_rule="mu", beta = 0,
                                                    sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                    verbose = False, return_costs = True, deterministic = True, seed=0)
        
        # Checking factors
        self.assertAlmostEqual(factors[0][0][0], 0.5488704375518113)
        self.assertAlmostEqual(factors[1][0][0], 0.9680879599528461)
        self.assertAlmostEqual(factors[2][0][0], 0.9650465314632987)
        self.assertAlmostEqual(core[0,0,0], 0.3744250029550508)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 0.01749656252808407)
        self.assertAlmostEqual(cost_fct_vals[-1], 0.014723505531139436)



# %% Run tests
if __name__ == '__main__':
    unittest.main()