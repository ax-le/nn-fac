# -*- coding: utf-8 -*-
import math
import unittest
import numpy as np
import nn_fac.update_rules.nnls as nnls
import nn_fac.utils.errors as err


def _synthetic_problem(m=20, n=10, r=5, seed=42):
    """Return (data, U, V_true, V_init) with all-nonneg entries."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(0, 1, (m, r))
    V_true = rng.uniform(0, 1, (r, n))
    data = U @ V_true + 0.01 * rng.uniform(0, 1, (m, n))
    V_init = rng.uniform(0, 1, (r, n))
    return data, U, V_true, V_init


# ── hals_nnls ──────────────────────────────────────────────────────────────────

class TestHalsNnls(unittest.TestCase):

    # input validation
    def test_1d_data_raises(self):
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls(np.ones(8), np.ones((8, 4)), np.ones((4, 6)))

    def test_1d_U_raises(self):
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls(np.ones((8, 6)), np.ones(8), np.ones((4, 6)))

    def test_1d_V_raises(self):
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls(np.ones((8, 6)), np.ones((8, 4)), np.ones(4))

    def test_shape_mismatch_data_U_rows(self):
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls(np.ones((8, 6)), np.ones((7, 4)), np.ones((4, 6)))

    def test_shape_mismatch_data_V_cols(self):
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls(np.ones((8, 6)), np.ones((8, 4)), np.ones((4, 5)))

    def test_shape_mismatch_U_V_inner(self):
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls(np.ones((8, 6)), np.ones((8, 4)), np.ones((3, 6)))

    # output properties
    def test_returns_triple(self):
        data, U, _, V_init = _synthetic_problem()
        self.assertEqual(len(nnls.hals_nnls(data, U, V_init)), 3)

    def test_output_shape(self):
        data, U, _, V_init = _synthetic_problem()
        V, _, _ = nnls.hals_nnls(data, U, V_init)
        self.assertEqual(V.shape, V_init.shape)

    def test_output_nonneg(self):
        data, U, _, V_init = _synthetic_problem()
        V, _, _ = nnls.hals_nnls(data, U, V_init)
        self.assertTrue(np.all(V >= 0))

    def test_does_not_mutate_inputs(self):
        data, U, _, V_init = _synthetic_problem()
        d0, u0, v0 = data.copy(), U.copy(), V_init.copy()
        nnls.hals_nnls(data, U, V_init)
        np.testing.assert_array_equal(data, d0)
        np.testing.assert_array_equal(U, u0)
        np.testing.assert_array_equal(V_init, v0)

    # convergence
    def test_reconstruction_improves(self):
        data, U, _, V_init = _synthetic_problem()
        resid_init = np.linalg.norm(data - U @ V_init)
        V, _, _ = nnls.hals_nnls(data, U, V_init)
        self.assertLess(np.linalg.norm(data - U @ V), resid_init)

    # options
    def test_nonzero_raises_on_zero_column(self):
        rng = np.random.default_rng(0)
        U = rng.uniform(0, 1, (8, 4))
        U[:, 2] = 0
        data = rng.uniform(0, 1, (8, 6))
        V_init = rng.uniform(0, 1, (4, 6))
        with self.assertRaises(err.ZeroColumnWhenUnautorized):
            nnls.hals_nnls(data, U, V_init, nonzero=True)

    def test_zero_column_skipped_without_nonzero(self):
        rng = np.random.default_rng(0)
        U = rng.uniform(0, 1, (8, 4))
        U[:, 2] = 0
        data = rng.uniform(0, 1, (8, 6))
        V_init = rng.uniform(0, 1, (4, 6))
        V, _, _ = nnls.hals_nnls(data, U, V_init, nonzero=False)
        self.assertTrue(np.all(V >= 0))

    def test_normalize_produces_unit_rows(self):
        data, U, _, V_init = _synthetic_problem()
        V, _, _ = nnls.hals_nnls(data, U, V_init, normalize=True)
        np.testing.assert_allclose(np.linalg.norm(V, axis=1), np.ones(V.shape[0]), atol=1e-10)

    def test_sparsity_increases_zeros(self):
        data, U, _, V_init = _synthetic_problem(seed=0)
        V_dense, _, _ = nnls.hals_nnls(data, U, V_init.copy())
        V_sparse, _, _ = nnls.hals_nnls(data, U, V_init.copy(), sparsity_coefficient=0.5)
        self.assertGreaterEqual(np.sum(V_sparse == 0), np.sum(V_dense == 0))


# ── hals_nnls_acc ──────────────────────────────────────────────────────────────

class TestHalsNnlsAcc(unittest.TestCase):

    # input validation — mirrors hals_nnls exactly
    def test_1d_data_raises(self):
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls_acc(np.ones(8), np.ones((8, 4)), np.ones((4, 6)))

    def test_1d_U_raises(self):
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls_acc(np.ones((8, 6)), np.ones(8), np.ones((4, 6)))

    def test_1d_V_raises(self):
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls_acc(np.ones((8, 6)), np.ones((8, 4)), np.ones(4))

    def test_shape_mismatch_data_U_rows(self):
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls_acc(np.ones((8, 6)), np.ones((7, 4)), np.ones((4, 6)))

    def test_shape_mismatch_data_V_cols(self):
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls_acc(np.ones((8, 6)), np.ones((8, 4)), np.ones((4, 5)))

    def test_shape_mismatch_U_V_inner(self):
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls_acc(np.ones((8, 6)), np.ones((8, 4)), np.ones((3, 6)))

    # output properties
    def test_returns_triple(self):
        data, U, _, V_init = _synthetic_problem()
        self.assertEqual(len(nnls.hals_nnls_acc(data, U, V_init)), 3)

    def test_output_shape(self):
        data, U, _, V_init = _synthetic_problem()
        V, _, _ = nnls.hals_nnls_acc(data, U, V_init)
        self.assertEqual(V.shape, V_init.shape)

    def test_output_nonneg(self):
        data, U, _, V_init = _synthetic_problem()
        V, _, _ = nnls.hals_nnls_acc(data, U, V_init)
        self.assertTrue(np.all(V >= 0))

    def test_does_not_mutate_inputs(self):
        data, U, _, V_init = _synthetic_problem()
        d0, u0, v0 = data.copy(), U.copy(), V_init.copy()
        nnls.hals_nnls_acc(data, U, V_init)
        np.testing.assert_array_equal(data, d0)
        np.testing.assert_array_equal(U, u0)
        np.testing.assert_array_equal(V_init, v0)

    # convergence
    def test_reconstruction_improves(self):
        data, U, _, V_init = _synthetic_problem()
        resid_init = np.linalg.norm(data - U @ V_init)
        V, _, _ = nnls.hals_nnls_acc(data, U, V_init, alpha=math.inf)
        self.assertLess(np.linalg.norm(data - U @ V), resid_init)

    def test_residual_matches_non_acc(self):
        """Both solvers from the same init should reach essentially the same residual."""
        data, U, _, V_init = _synthetic_problem(seed=7)
        V1, _, _ = nnls.hals_nnls(data, U, V_init.copy(), maxiter=500)
        V2, _, _ = nnls.hals_nnls_acc(data, U, V_init.copy(), maxiter=500, alpha=math.inf, delta=1e-10)
        resid1 = np.linalg.norm(data - U @ V1)
        resid2 = np.linalg.norm(data - U @ V2)
        self.assertAlmostEqual(resid1, resid2, places=2)

    def test_cnt_bounded_by_maxiter(self):
        # cnt starts at 1, increments once after the last allowed iteration,
        # so the maximum possible value is maxiter + 1.
        data, U, _, V_init = _synthetic_problem()
        maxiter = 10
        _, _, cnt = nnls.hals_nnls_acc(data, U, V_init, maxiter=maxiter, alpha=math.inf, delta=0)
        self.assertLessEqual(cnt, maxiter + 1)

    # options
    def test_nonzero_raises_on_zero_column(self):
        rng = np.random.default_rng(0)
        U = rng.uniform(0, 1, (8, 4))
        U[:, 1] = 0
        data = rng.uniform(0, 1, (8, 6))
        V_init = rng.uniform(0, 1, (4, 6))
        with self.assertRaises(err.ZeroColumnWhenUnautorized):
            nnls.hals_nnls_acc(data, U, V_init, nonzero=True)

    def test_zero_column_skipped_without_nonzero(self):
        rng = np.random.default_rng(0)
        U = rng.uniform(0, 1, (8, 4))
        U[:, 1] = 0
        data = rng.uniform(0, 1, (8, 6))
        V_init = rng.uniform(0, 1, (4, 6))
        V, _, _ = nnls.hals_nnls_acc(data, U, V_init, nonzero=False)
        self.assertTrue(np.all(V >= 0))

    def test_normalize_produces_unit_rows(self):
        data, U, _, V_init = _synthetic_problem()
        V, _, _ = nnls.hals_nnls_acc(data, U, V_init, normalize=True, alpha=math.inf)
        np.testing.assert_allclose(np.linalg.norm(V, axis=1), np.ones(V.shape[0]), atol=1e-10)

    def test_sparsity_increases_zeros(self):
        data, U, _, V_init = _synthetic_problem(seed=0)
        V_dense, _, _ = nnls.hals_nnls_acc(data, U, V_init.copy(), alpha=math.inf)
        V_sparse, _, _ = nnls.hals_nnls_acc(data, U, V_init.copy(), sparsity_coefficient=0.5, alpha=math.inf)
        self.assertGreaterEqual(np.sum(V_sparse == 0), np.sum(V_dense == 0))


# ── switch_alternate_hals ──────────────────────────────────────────────────────

class TestSwitchAlternateHals(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(99)
        self.m, self.r, self.n = 20, 5, 15
        self.U = rng.uniform(0, 1, (self.m, self.r))
        self.V = rng.uniform(0, 1, (self.r, self.n))
        self.data = rng.uniform(0, 1, (self.m, self.n))

    def test_V_update_shape(self):
        self.assertEqual(nnls.switch_alternate_hals(self.data, self.U, self.V, "V").shape, self.V.shape)

    def test_H_update_shape(self):
        self.assertEqual(nnls.switch_alternate_hals(self.data, self.U, self.V, "H").shape, self.V.shape)

    def test_U_update_shape(self):
        self.assertEqual(nnls.switch_alternate_hals(self.data, self.U, self.V, "U").shape, self.U.shape)

    def test_W_update_shape(self):
        self.assertEqual(nnls.switch_alternate_hals(self.data, self.U, self.V, "W").shape, self.U.shape)

    def test_invalid_matrix_raises(self):
        with self.assertRaises(err.InvalidArgumentValue):
            nnls.switch_alternate_hals(self.data, self.U, self.V, "X")

    def test_V_output_nonneg(self):
        self.assertTrue(np.all(nnls.switch_alternate_hals(self.data, self.U, self.V, "V") >= 0))

    def test_U_output_nonneg(self):
        self.assertTrue(np.all(nnls.switch_alternate_hals(self.data, self.U, self.V, "U") >= 0))

    def test_V_matches_direct_hals_nnls(self):
        """'V' dispatch must equal calling hals_nnls directly."""
        V_switch = nnls.switch_alternate_hals(self.data, self.U, self.V, "V")
        V_direct, _, _ = nnls.hals_nnls(self.data, self.U, self.V)
        np.testing.assert_allclose(V_switch, V_direct, atol=1e-10)

    def test_U_matches_transposed_hals_nnls(self):
        """'U' dispatch must equal hals_nnls on the transposed problem."""
        U_switch = nnls.switch_alternate_hals(self.data, self.U, self.V, "U")
        UtT, _, _ = nnls.hals_nnls(self.data.T, self.V.T, self.U.T)
        np.testing.assert_allclose(U_switch, UtT.T, atol=1e-10)

    def test_H_same_as_V(self):
        V = nnls.switch_alternate_hals(self.data, self.U, self.V, "V")
        H = nnls.switch_alternate_hals(self.data, self.U, self.V, "H")
        np.testing.assert_array_equal(V, H)

    def test_W_same_as_U(self):
        U = nnls.switch_alternate_hals(self.data, self.U, self.V, "U")
        W = nnls.switch_alternate_hals(self.data, self.U, self.V, "W")
        np.testing.assert_array_equal(U, W)


# ── switch_alternate_hals_acc ──────────────────────────────────────────────────

class TestSwitchAlternateHalsAcc(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(99)
        self.m, self.r, self.n = 20, 5, 15
        self.U = rng.uniform(0, 1, (self.m, self.r))
        self.V = rng.uniform(0, 1, (self.r, self.n))
        self.data = rng.uniform(0, 1, (self.m, self.n))

    def test_V_update_shape(self):
        self.assertEqual(nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "V").shape, self.V.shape)

    def test_H_update_shape(self):
        self.assertEqual(nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "H").shape, self.V.shape)

    def test_U_update_shape(self):
        self.assertEqual(nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "U").shape, self.U.shape)

    def test_W_update_shape(self):
        self.assertEqual(nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "W").shape, self.U.shape)

    def test_invalid_matrix_raises(self):
        with self.assertRaises(err.InvalidArgumentValue):
            nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "Z")

    def test_V_output_nonneg(self):
        self.assertTrue(np.all(nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "V") >= 0))

    def test_U_output_nonneg(self):
        self.assertTrue(np.all(nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "U") >= 0))

    def test_deterministic_is_reproducible(self):
        V1 = nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "V", deterministic=True)
        V2 = nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "V", deterministic=True)
        np.testing.assert_array_equal(V1, V2)

    def test_V_matches_direct_hals_nnls_acc(self):
        """'V' dispatch (deterministic) must equal hals_nnls_acc directly."""
        maxiter, delta = 100, 0.01
        V_switch = nnls.switch_alternate_hals_acc(
            self.data, self.U, self.V, "V", deterministic=True, maxiter=maxiter, delta=delta)
        V_direct, _, _ = nnls.hals_nnls_acc(
            self.data, self.U, self.V, alpha=math.inf, maxiter=maxiter, delta=delta)
        np.testing.assert_allclose(V_switch, V_direct, atol=1e-10)

    def test_U_matches_transposed_hals_nnls_acc(self):
        """'U' dispatch (deterministic) must equal hals_nnls_acc on the transposed problem."""
        maxiter, delta = 100, 0.01
        U_switch = nnls.switch_alternate_hals_acc(
            self.data, self.U, self.V, "U", deterministic=True, maxiter=maxiter, delta=delta)
        UtT, _, _ = nnls.hals_nnls_acc(
            self.data.T, self.V.T, self.U.T, alpha=math.inf, maxiter=maxiter, delta=delta)
        np.testing.assert_allclose(U_switch, UtT.T, atol=1e-10)

    def test_H_same_as_V(self):
        V = nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "V", deterministic=True)
        H = nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "H", deterministic=True)
        np.testing.assert_array_equal(V, H)

    def test_W_same_as_U(self):
        U = nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "U", deterministic=True)
        W = nnls.switch_alternate_hals_acc(self.data, self.U, self.V, "W", deterministic=True)
        np.testing.assert_array_equal(U, W)


if __name__ == '__main__':
    unittest.main()
