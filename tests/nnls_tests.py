# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:47:27 2020

@author: amarmore
"""
# Tests on development of the NTD algorithm.

import unittest
import random
import numpy as np
import nn_fac.nnls as nnls
import nn_fac.errors as err

class NnlsTests(unittest.TestCase):

    def test_wrong_arguments(self):
        """
        Verifies that errors are raised when necessary.
        """
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls_acc(np.random.random((8,8)), np.random.random((8,8)), np.array([]))

        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls_acc(np.random.random((8)), np.random.random((8,8)), np.random.random((8,8)))
        
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls_acc(np.random.random((8,8)), np.random.random((8)), np.random.random((8,8)))

    def test_error_in_optim(self):
        """
        Verifies that errors are raised when necessary.
        """
        UtU = np.random.random((8,8))
        UtU[2,2] = 0
        nnls.hals_nnls_acc(np.random.random((8,8)), UtU, np.random.random((8,8)))
        with self.assertRaises(err.ZeroColumnWhenUnautorized):
            nnls.hals_nnls_acc(np.random.random((8,8)), UtU, np.random.random((8,8)), nonzero = True)
    
    def test_nnls_for_a_vector(self):
        """
        Verifies a question raised by Jeremy: is the nnls working with vectors as input ?
        """
        UtU = np.random.random((15,15))
        nnls.hals_nnls_acc(np.random.random((8,1)), UtU, np.random.random((15,1)))
        with self.assertRaises(err.ArgumentException):
            nnls.hals_nnls_acc(np.random.random((8)), UtU, np.random.random((15,1)), nonzero = True)
 

# %% Run tests
if __name__ == '__main__':
    unittest.main()