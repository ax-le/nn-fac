# -*- coding: utf-8 -*-
"""
Tests on development of the NTF algorithm.
"""
# Tests on development of the NTF algorithm.

import unittest
import tensorly as tl
from tensorly.cp_tensor import cp_to_tensor
import random
import numpy as np
from nn_fac.ntf import ntf
import nn_fac.utils.errors as err

class NTFTests(unittest.TestCase):

    def setUp(self):
        """
        setUp function, not to redefine the objects in each test function.
        """
        np.random.seed(0)
        random.seed(0)

        self.random_rank = random.randint(3, 10)
        self.random_shape_tens = (random.randint(20, 60), random.randint(20, 60), random.randint(20, 60))

        self.factors_0 = [np.random.rand(shape_mode, self.random_rank) for shape_mode in self.random_shape_tens]
        self.init_by_product_tensor = tl.tensor(cp_to_tensor((None, self.factors_0)))

    # %% Normal computation
    def test_invalid_init_values(self):
        with self.assertRaises(err.InvalidInitializationType):
            factors, errs, toc = ntf(self.init_by_product_tensor, self.random_rank, init = "string", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None], normalize = [False, False, False])

    def test_invalid_custom_init_values(self):
        with self.assertRaises(err.CustomNotEngouhFactors):
            factors, errs, toc = ntf(self.init_by_product_tensor, self.random_rank, init = "custom", factors_0 = [self.factors_0[0], self.factors_0[1]],
                                    return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None], normalize = [False, False, False])

        with self.assertRaises(err.CustomNotValidFactors):
            factors, errs, toc = ntf(self.init_by_product_tensor, self.random_rank, init = "custom", factors_0 = [self.factors_0[0], self.factors_0[1], None],
                                    return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None], normalize = [False, False, False])

    def test_invalid_update_rule(self):
        with self.assertRaises(err.InvalidArgumentValue):
            ntf(self.init_by_product_tensor, self.random_rank, update_rule = "bogus",
                sparsity_coefficients = [None, None, None], normalize = [False, False, False])

    def test_invalid_hals_beta(self):
        with self.assertRaises(err.InvalidArgumentValue):
            ntf(self.init_by_product_tensor, self.random_rank, update_rule = "hals", beta = 1,
                sparsity_coefficients = [None, None, None], normalize = [False, False, False])

    # %% Check that update rules indeed decrease the loss
    def strictly_decreasing(self, L):
        return all(x>y for x, y in zip(L, L[1:]))

    def test_good_random_decomp_hals(self):
        factors, errs, toc = ntf(self.init_by_product_tensor, self.random_rank, init = "random", n_iter_max = 100, tol = 1e-14,
                                    update_rule = "hals", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None], normalize = [False, False, False])
        self.assertAlmostEqual(errs[-1], 0, places = 2)
        self.assertTrue(self.strictly_decreasing(errs))

    def test_good_nndsvd_decomp_hals(self):
        factors, errs, toc = ntf(self.init_by_product_tensor, self.random_rank, init = "nndsvd", n_iter_max = 100, tol = 1e-14,
                                    update_rule = "hals", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None], normalize = [False, False, False])
        self.assertAlmostEqual(errs[-1], 0, places = 2)
        self.assertTrue(self.strictly_decreasing(errs))

    def test_good_custom_decomp(self):
        factors, errs, toc = ntf(self.init_by_product_tensor + 1e-2 * np.random.rand(*self.random_shape_tens), self.random_rank,
                                    init = "custom", factors_0 = [factor.copy() for factor in self.factors_0],
                                    n_iter_max = 100, tol = 1e-14, update_rule = "hals", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None], normalize = [False, False, False])
        self.assertAlmostEqual(errs[-1], 0, places = 2)
        self.assertTrue(self.strictly_decreasing(errs))

    def test_good_random_decomp_mu(self):
        for beta in [0, 1, 2]:
            with self.subTest(beta = beta):
                factors, errs, toc = ntf(self.init_by_product_tensor, self.random_rank, init = "random", n_iter_max = 100, tol = 1e-14,
                                        update_rule = "mu", beta = beta, return_costs = True, verbose = False,
                                        sparsity_coefficients = [None, None, None], normalize = [False, False, False])
                self.assertLess(errs[-1], errs[0])

    # %% fixed_modes: fixing a factor should leave it untouched
    # Note: factors_0 perturbed with noise, since starting from the exact
    # ground-truth factors is already a fixed point of the HALS update (no
    # actual update would happen for *any* mode, defeating the test).
    def test_fixed_mode_is_not_updated(self):
        factors_init = [factor + 0.1 * np.random.rand(*factor.shape) for factor in self.factors_0]
        factors, errs, toc = ntf(self.init_by_product_tensor, self.random_rank, init = "custom",
                                    factors_0 = [factor.copy() for factor in factors_init],
                                    fixed_modes = [True, False, False], n_iter_max = 10, tol = 1e-14,
                                    update_rule = "hals", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None], normalize = [False, False, False])
        np.testing.assert_allclose(factors[0], factors_init[0])
        self.assertFalse(np.allclose(factors[1], factors_init[1]))
        self.assertFalse(np.allclose(factors[2], factors_init[2]))

    def test_default_fixed_modes_updates_every_factor(self):
        factors_init = [factor + 0.1 * np.random.rand(*factor.shape) for factor in self.factors_0]
        factors, errs, toc = ntf(self.init_by_product_tensor, self.random_rank, init = "custom",
                                    factors_0 = [factor.copy() for factor in factors_init],
                                    n_iter_max = 5, tol = 1e-14, update_rule = "hals", return_costs = True, verbose = False,
                                    sparsity_coefficients = [None, None, None], normalize = [False, False, False])
        for mode in range(3):
            self.assertFalse(np.allclose(factors[mode], factors_init[mode]))

    # %% Returned factors should stay a plain list of arrays, even for non-cubic tensors
    def test_returns_list_of_factors_for_non_cubic_tensor(self):
        self.assertNotEqual(self.random_shape_tens[0], self.random_shape_tens[1])
        factors = ntf(self.init_by_product_tensor, self.random_rank, init = "random", n_iter_max = 5,
                                    sparsity_coefficients = [None, None, None], normalize = [False, False, False])
        self.assertEqual(len(factors), 3)
        for mode, factor in enumerate(factors):
            self.assertEqual(factor.shape, (self.random_shape_tens[mode], self.random_rank))


# %% Run tests
if __name__ == '__main__':
    unittest.main()
