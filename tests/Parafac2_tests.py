# -*- coding: utf-8 -*-
"""
Tests on development of the PARAFAC2 algorithm.
"""

import unittest
import random
import numpy as np
from nn_fac.parafac2 import parafac_2, compute_parafac_2
import nn_fac.utils.errors as err

class Parafac2Tests(unittest.TestCase):

    def setUp(self):
        """
        setUp function, not to redefine the objects in each test function.
        """
        np.random.seed(0)
        random.seed(0)

        self.rank = random.randint(3, 6)
        self.nb_rows = random.randint(10, 20)
        self.nb_columns = random.randint(10, 20)
        self.nb_channel = random.randint(3, 6)

        self.H_true = np.random.rand(self.rank, self.nb_columns) + 0.1
        self.W_list_true = [np.random.rand(self.nb_rows, self.rank) + 0.1 for _ in range(self.nb_channel)]
        self.D_list_true = [np.diag(np.random.rand(self.rank) + 0.2) for _ in range(self.nb_channel)]
        self.slices = [self.W_list_true[k] @ self.D_list_true[k] @ self.H_true for k in range(self.nb_channel)]

        # Perturbed initialization, used whenever the test needs an initial
        # guess that is not already (close to) a fixed point of the updates.
        self.W_list_init = [w + 0.1 * np.random.rand(*w.shape) for w in self.W_list_true]
        self.D_list_init = [d.copy() for d in self.D_list_true]
        self.H_init = self.H_true + 0.1 * np.random.rand(*self.H_true.shape)
        self.P_list_init = [np.identity(self.nb_rows)[:, 0:self.rank] for _ in range(self.nb_channel)]

    # %% Argument / initialization validation
    def test_invalid_init_values(self):
        with self.assertRaises(err.InvalidInitializationType):
            parafac_2(self.slices, self.rank, init_with_P = True, init = "string", n_iter_max = 2)

    def test_invalid_custom_init_values(self):
        with self.assertRaises(err.CustomNotValidFactors):
            parafac_2(self.slices, self.rank, init_with_P = True, init = "custom", n_iter_max = 2)

    def test_initialization_not_valid_when_missing_coupling_factor(self):
        with self.assertRaises(err.InitializationNotValid):
            compute_parafac_2(self.slices, self.rank, W_list_in = self.W_list_init, H_0 = self.H_init.copy(),
                                D_list_in = self.D_list_init, init_with_P = True, W_star_in = None, P_list_in = None,
                                n_iter_max = 2)

    # %% Check that the cost function substantially decreases
    def test_good_decomp_random_init_with_P(self):
        W_list, H, D_list, costs, toc = parafac_2(self.slices, self.rank, init_with_P = True, init = "random",
                                                    n_iter_max = 150, tol = 1e-14, return_costs = True,
                                                    deterministic = True, seed = 0)
        self.assertLess(costs[-1], 0.3 * costs[0])

    def test_good_decomp_random_init_with_W_star(self):
        W_list, H, D_list, costs, toc = parafac_2(self.slices, self.rank, init_with_P = False, init = "random",
                                                    n_iter_max = 150, tol = 1e-14, return_costs = True,
                                                    deterministic = True, seed = 0)
        self.assertLess(costs[-1], 0.3 * costs[0])

    def test_good_decomp_nndsvd_init(self):
        W_list, H, D_list, costs, toc = parafac_2(self.slices, self.rank, init_with_P = True, init = "nndsvd",
                                                    n_iter_max = 150, tol = 1e-14, return_costs = True,
                                                    deterministic = True, seed = 0)
        self.assertLess(costs[-1], 0.3 * costs[0])

    # %% Determinism
    def test_light_determinism(self):
        first_try = parafac_2(self.slices, self.rank, init_with_P = True, init = "random", n_iter_max = 20,
                                tol = 1e-14, return_costs = True, deterministic = True, seed = 0)
        second_try = parafac_2(self.slices, self.rank, init_with_P = True, init = "random", n_iter_max = 20,
                                tol = 1e-14, return_costs = True, deterministic = True, seed = 0)
        self.assertEqual(first_try[3][-1], second_try[3][-1])

    # %% fixed_modes: fixing a factor should leave it untouched
    def test_fixed_W_list_is_not_updated(self):
        W_list, H, D_list, costs, toc = parafac_2(self.slices, self.rank, init_with_P = True, init = "custom",
                                W_list_in = [w.copy() for w in self.W_list_init], H = self.H_init.copy(),
                                D_list_in = [d.copy() for d in self.D_list_init], P_list = self.P_list_init,
                                fixed_modes = [True, False, False, False, False],
                                n_iter_max = 5, tol = 1e-14, return_costs = True, deterministic = True, seed = 0)
        for k in range(self.nb_channel):
            np.testing.assert_allclose(W_list[k], self.W_list_init[k])

    def test_fixed_H_is_not_updated(self):
        W_list, H, D_list, costs, toc = parafac_2(self.slices, self.rank, init_with_P = True, init = "custom",
                                W_list_in = [w.copy() for w in self.W_list_init], H = self.H_init.copy(),
                                D_list_in = [d.copy() for d in self.D_list_init], P_list = self.P_list_init,
                                fixed_modes = [False, True, False, False, False],
                                n_iter_max = 5, tol = 1e-14, return_costs = True, deterministic = True, seed = 0)
        np.testing.assert_allclose(H, self.H_init)

    def test_fixed_D_list_is_not_updated(self):
        W_list, H, D_list, costs, toc = parafac_2(self.slices, self.rank, init_with_P = True, init = "custom",
                                W_list_in = [w.copy() for w in self.W_list_init], H = self.H_init.copy(),
                                D_list_in = [d.copy() for d in self.D_list_init], P_list = self.P_list_init,
                                fixed_modes = [False, False, True, False, False],
                                n_iter_max = 5, tol = 1e-14, return_costs = True, deterministic = True, seed = 0)
        for k in range(self.nb_channel):
            np.testing.assert_allclose(D_list[k], self.D_list_init[k])

    def test_default_fixed_modes_updates_every_factor(self):
        W_list, H, D_list, costs, toc = parafac_2(self.slices, self.rank, init_with_P = True, init = "custom",
                                W_list_in = [w.copy() for w in self.W_list_init], H = self.H_init.copy(),
                                D_list_in = [d.copy() for d in self.D_list_init], P_list = self.P_list_init,
                                n_iter_max = 5, tol = 1e-14, return_costs = True, deterministic = True, seed = 0)
        self.assertFalse(all(np.allclose(W_list[k], self.W_list_init[k]) for k in range(self.nb_channel)))
        self.assertFalse(np.allclose(H, self.H_init))
        self.assertFalse(all(np.allclose(D_list[k], self.D_list_init[k]) for k in range(self.nb_channel)))


# %% Run tests
if __name__ == '__main__':
    unittest.main()
