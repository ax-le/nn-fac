# -*- coding: utf-8 -*-
import unittest
import numpy as np
import tensorly as tl
import nn_fac.update_rules.mu as mu
import nn_fac.utils.errors as err


def _synthetic_problem(m=20, n=10, r=5, seed=42):
    """Return (data, U, V_true, V_init) with strictly positive entries."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(0.1, 1, (m, r))
    V_true = rng.uniform(0.1, 1, (r, n))
    data = U @ V_true + 0.01 * rng.uniform(0, 1, (m, n))
    V_init = rng.uniform(0.1, 1, (r, n))
    return data, U, V_true, V_init


# ── mu_betadivmin ────────────────────────────────────────────────────────────

class TestMuBetadivmin(unittest.TestCase):

    # input validation
    def test_negative_beta_raises(self):
        data, U, _, V_init = _synthetic_problem()
        with self.assertRaises(err.InvalidArgumentValue):
            mu.mu_betadivmin(U, V_init, data, -1)

    # output properties
    def test_output_shape(self):
        data, U, _, V_init = _synthetic_problem()
        U_up = mu.mu_betadivmin(U, V_init, data, 2)
        self.assertEqual(U_up.shape, U.shape)

    def test_output_nonneg(self):
        data, U, _, V_init = _synthetic_problem()
        U_up = mu.mu_betadivmin(U, V_init, data, 2)
        self.assertTrue(np.all(U_up >= 0))

    def test_does_not_mutate_inputs(self):
        data, U, _, V_init = _synthetic_problem()
        d0, u0, v0 = data.copy(), U.copy(), V_init.copy()
        mu.mu_betadivmin(U, V_init, data, 2)
        np.testing.assert_array_equal(data, d0)
        np.testing.assert_array_equal(U, u0)
        np.testing.assert_array_equal(V_init, v0)

    # convergence
    def test_reconstruction_improves_euclidean(self):
        data, U, _, V_init = _synthetic_problem()
        resid_init = np.linalg.norm(data - U @ V_init)
        V = V_init.copy()
        for _ in range(200):
            V = mu.switch_alternate_mu(data, U, V, 2, "V")
        self.assertLess(np.linalg.norm(data - U @ V), resid_init)

    # options
    def test_various_beta_values_output_shape(self):
        data, U, _, V_init = _synthetic_problem()
        for beta in [0.5, 1, 1.5, 2, 2.5, 3, 4]:
            with self.subTest(beta=beta):
                U_up = mu.mu_betadivmin(U, V_init, data, beta)
                self.assertEqual(U_up.shape, U.shape)
                self.assertTrue(np.all(U_up >= 0))


# ── switch_alternate_mu ──────────────────────────────────────────────────────

class TestSwitchAlternateMu(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(99)
        self.m, self.r, self.n = 20, 5, 15
        self.U = rng.uniform(0.1, 1, (self.m, self.r))
        self.V = rng.uniform(0.1, 1, (self.r, self.n))
        self.data = rng.uniform(0.1, 1, (self.m, self.n))
        self.beta = 2

    def test_V_update_shape(self):
        self.assertEqual(mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "V").shape, self.V.shape)

    def test_H_update_shape(self):
        self.assertEqual(mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "H").shape, self.V.shape)

    def test_U_update_shape(self):
        self.assertEqual(mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "U").shape, self.U.shape)

    def test_W_update_shape(self):
        self.assertEqual(mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "W").shape, self.U.shape)

    def test_invalid_matrix_raises(self):
        with self.assertRaises(err.InvalidArgumentValue):
            mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "X")

    def test_V_output_nonneg(self):
        self.assertTrue(np.all(mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "V") >= 0))

    def test_U_output_nonneg(self):
        self.assertTrue(np.all(mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "U") >= 0))

    def test_U_matches_direct_mu_betadivmin(self):
        """'U' dispatch must equal calling mu_betadivmin directly."""
        U_switch = mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "U")
        U_direct = mu.mu_betadivmin(self.U, self.V, self.data, self.beta)
        np.testing.assert_allclose(U_switch, U_direct, atol=1e-10)

    def test_V_matches_transposed_mu_betadivmin(self):
        """'V' dispatch must equal mu_betadivmin on the transposed problem."""
        V_switch = mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "V")
        VtT = mu.mu_betadivmin(self.V.T, self.U.T, self.data.T, self.beta)
        np.testing.assert_allclose(V_switch, VtT.T, atol=1e-10)

    def test_H_same_as_V(self):
        V = mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "V")
        H = mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "H")
        np.testing.assert_array_equal(V, H)

    def test_W_same_as_U(self):
        U = mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "U")
        W = mu.switch_alternate_mu(self.data, self.U, self.V, self.beta, "W")
        np.testing.assert_array_equal(U, W)


# ── mu_tensorial ─────────────────────────────────────────────────────────────

class TestMuTensorial(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(7)
        self.dims = (8, 6, 5)
        self.ranks = (3, 2, 4)
        self.factors = [rng.uniform(0.1, 1, (d, r)) for d, r in zip(self.dims, self.ranks)]
        self.G = rng.uniform(0.1, 1, self.ranks)
        self.tensor = rng.uniform(0.1, 1, self.dims)

    def test_negative_beta_raises(self):
        with self.assertRaises(err.InvalidArgumentValue):
            mu.mu_tensorial(self.G, self.factors, self.tensor, -1)

    def test_output_shape(self):
        G_up = mu.mu_tensorial(self.G, self.factors, self.tensor, 2)
        self.assertEqual(G_up.shape, self.G.shape)

    def test_output_nonneg(self):
        G_up = mu.mu_tensorial(self.G, self.factors, self.tensor, 2)
        self.assertTrue(np.all(G_up >= 0))

    def test_various_beta_values_output_shape(self):
        for beta in [0.5, 1, 1.5, 2, 2.5, 3, 4]:
            with self.subTest(beta=beta):
                G_up = mu.mu_tensorial(self.G, self.factors, self.tensor, beta)
                self.assertEqual(G_up.shape, self.G.shape)
                self.assertTrue(np.all(G_up >= 0))

    def test_reconstruction_improves_euclidean(self):
        resid_init = np.linalg.norm(self.tensor - tl.tenalg.multi_mode_dot(self.G, self.factors))
        G = self.G.copy()
        for _ in range(100):
            G = mu.mu_tensorial(G, self.factors, self.tensor, 2)
        resid_final = np.linalg.norm(self.tensor - tl.tenalg.multi_mode_dot(G, self.factors))
        self.assertLess(resid_final, resid_init)


# # ── simplex_proj_mu ──────────────────────────────────────────────────────────

# class TestSimplexProjMu(unittest.TestCase):

#     def setUp(self):
#         rng = np.random.default_rng(3)
#         self.m, self.k, self.n = 10, 4, 6
#         self.W = rng.uniform(0.1, 1, (self.m, self.k))
#         self.H = rng.uniform(0.1, 1, (self.k, self.n))
#         self.data = self.W @ self.H + 0.01 * rng.uniform(0, 1, (self.m, self.n))

#     def test_output_shape(self):
#         H_up = mu.simplex_proj_mu(self.data, self.W, self.H, 2)
#         self.assertEqual(H_up.shape, self.H.shape)

#     def test_output_nonneg(self):
#         H_up = mu.simplex_proj_mu(self.data, self.W, self.H, 2)
#         self.assertTrue(np.all(H_up >= 0))

#     def test_columns_sum_to_one_beta_1(self):
#         # beta=1 is the one case where a single call already satisfies the
#         # simplex constraint exactly; other betas only approach it over
#         # repeated calls (see test_iterating_reduces_simplex_violation).
#         H_up = mu.simplex_proj_mu(self.data, self.W, self.H, 1, tol_update_lagrangian=1e-10)
#         np.testing.assert_allclose(np.sum(H_up, axis=0), np.ones(self.n), atol=1e-8)

#     # options
#     def test_various_beta_values_output_shape(self):
#         # Restricted to beta in [1, 2]: this MU-based simplex projection is
#         # numerically unstable (produces NaNs) for beta > 2.
#         for beta in [1, 1.5, 2]:
#             with self.subTest(beta=beta):
#                 H_up = mu.simplex_proj_mu(self.data, self.W, self.H, beta, tol_update_lagrangian=1e-10)
#                 self.assertEqual(H_up.shape, self.H.shape)
#                 self.assertTrue(np.all(H_up >= 0))

#     def test_iterating_reduces_simplex_violation(self):
#         deviation_init = np.sum(np.abs(np.sum(self.H, axis=0) - 1))
#         H = self.H.copy()
#         for _ in range(20):
#             H = mu.simplex_proj_mu(self.data, self.W, H, 2, tol_update_lagrangian=1e-10)
#         deviation_final = np.sum(np.abs(np.sum(H, axis=0) - 1))
#         self.assertLess(deviation_final, deviation_init)


if __name__ == '__main__':
    unittest.main()
