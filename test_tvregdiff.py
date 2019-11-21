"""
Unit tests for tvregdiff.py which test if this Python
implementation of TVREGDiff replicates the MATLAB demo
scipt outputs from Rick Chartrand's webpage:
https://sites.google.com/site/dnartrahckcir/home/tvdiff-code
"""

import os
import unittest
import numpy as np
from numpy.testing import assert_allclose
from tvregdiff import TVRegDiff


class TVDiffTest(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'test_data'

    def test_case_small(self):
        """Small-scale example from paper Rick Chartrand,
        "Numerical differentiation of noisy, nonsmooth data," ISRN
        Applied Mathematics, Vol. 2011, Article ID 164564, 2011.
        """

        # Load data (data is from smalldemodata.mat)
        noisy_abs_data = np.loadtxt('smalldemodata.csv')
        self.assertEqual(noisy_abs_data.shape, (100,))

        # Test with one iteration
        n_iters = 1
        alph = 0.2
        scale = 'small'
        ep = 1e-6
        dx =  0.01
        u = TVRegDiff(noisy_abs_data, n_iters, alph, u0=None, scale=scale, 
                      ep=ep, dx=dx, plotflag=False, diagflag=True)
        self.assertEqual(u.shape, (101,))
        filepath = os.path.join(self.data_dir, 'smalldemo_u1.csv')
        u_test = np.loadtxt(filepath)
        assert_allclose(u, u_test)

        # Test with 500 iterations
        n_iters = 500
        u = TVRegDiff(noisy_abs_data, n_iters, alph, u0=None, scale=scale, 
                      ep=ep, dx=dx, plotflag=False, diagflag=True)
        self.assertEqual(u.shape, (101,))
        filepath = os.path.join(self.data_dir, 'smalldemo_u.csv')
        u_test = np.loadtxt(filepath)
        assert_allclose(u, u_test)


if __name__ == '__main__':
    unittest.main()
