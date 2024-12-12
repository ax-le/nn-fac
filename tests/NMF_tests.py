# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:47:27 2020

@author: amarmore
"""
# Tests on development of the NTD algorithm.

import unittest
import random
import numpy as np
from nn_fac.nmf import nmf
import nn_fac.utils.errors as err

class NMFTests(unittest.TestCase):

    def setUp(self):
        """
        setUp function, not to redefine the objects in each test function.
        """
        np.random.seed(0)
        random.seed(0)
       
        self.random_rank = random.randint(3,10)
        self.random_shape = (random.randint(20,100), random.randint(20,100))

        self.U_0 = np.random.rand(self.random_shape[0], self.random_rank)
        self.V_0 = np.random.rand(self.random_rank, self.random_shape[1])
        self.data_init = self.U_0@self.V_0 + 1e-2 * np.random.rand(self.random_shape[0], self.random_shape[1])

    # %% Testing a decomposition
    def test_decomposition_hals(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.data_init[0][0], 2.143518599859098)

        U, V, cost_fct_vals, toc = nmf(self.data_init, self.random_rank, init = "random", U_0 = None, V_0 = None, n_iter_max=10, tol=1e-8,
                                        update_rule = "hals", beta = 2,
                                        sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False],
                                        verbose=False, return_costs=True, deterministic=True)
        
        # Checking factors
        self.assertAlmostEqual(U[0][0], 0.36681975980591014)
        self.assertAlmostEqual(V[0][0], 1.3067452788067433)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 66.88690297118416)
        self.assertAlmostEqual(cost_fct_vals[-1], 1.3318257972164844)

    def test_decomposition_mu_beta2(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.data_init[0][0], 2.143518599859098)

        U, V, cost_fct_vals, toc = nmf(self.data_init, self.random_rank, init = "random", U_0 = None, V_0 = None, n_iter_max=10, tol=1e-8,
                                        update_rule = "mu", beta = 2,
                                        sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False],
                                        verbose=False, return_costs=True, deterministic=True)
        
        # Checking factors
        self.assertAlmostEqual(U[0][0], 0.35280947364767296)
        self.assertAlmostEqual(V[0][0], 0.44719984549809116)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 111.43110252634743)
        self.assertAlmostEqual(cost_fct_vals[-1], 68.8373870926001)

    def test_decomposition_mu_beta1(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.data_init[0][0], 2.143518599859098)

        U, V, cost_fct_vals, toc = nmf(self.data_init, self.random_rank, init = "random", U_0 = None, V_0 = None, n_iter_max=10, tol=1e-8,
                                        update_rule = "mu", beta = 1,
                                        sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False],
                                        verbose=False, return_costs=True, deterministic=True)
        
        # Checking factors
        self.assertAlmostEqual(U[0][0], 0.3718053134990678)
        self.assertAlmostEqual(V[0][0], 0.4367362187193684)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 51.47596084683006)
        self.assertAlmostEqual(cost_fct_vals[-1], 32.742423893466851)

    def test_decomposition_mu_beta0(self):

        # If this fails, determinism has failed. Hence, the rest of the test is bound to fail.
        self.assertAlmostEqual(self.data_init[0][0], 2.143518599859098)

        U, V, cost_fct_vals, toc = nmf(self.data_init, self.random_rank, init = "random", U_0 = None, V_0 = None, n_iter_max=10, tol=1e-8,
                                        update_rule = "mu", beta = 0,
                                        sparsity_coefficients = [None, None], fixed_modes = [], normalize = [False, False],
                                        verbose=False, return_costs=True, deterministic=True)
        
        # Checking factors
        self.assertAlmostEqual(U[0][0], 0.32746152037135323)
        self.assertAlmostEqual(V[0][0], 0.4098870587115991)
        
        # Checking errors
        self.assertAlmostEqual(cost_fct_vals[0], 71.40741383137126)
        self.assertAlmostEqual(cost_fct_vals[-1], 20.041539547898314)

# %% Run tests
if __name__ == '__main__':
    unittest.main()