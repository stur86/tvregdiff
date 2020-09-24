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
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = False

# Functions to test on
functions = {
    'abs': [np.abs, np.sign],
    'sigmoid': [lambda x: np.exp(x)/(1+np.exp(x)),
                lambda x: np.exp(x)/(1+np.exp(x))-np.exp(2*x)/(1+np.exp(x))**2]
}


def rms(x):
    return np.average(x**2)**0.5


class TVDiffTest(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'test_data'
        self.longMessage = True

    def test_small(self):

        n = 50
        x = np.linspace(-5, 5, n)
        dx = x[1]-x[0]

        for fname, (f, df) in functions.items():

            data = f(x) + (np.random.random(n)-0.5)*0.05
            targ = df(x)

            u1a = TVRegDiff(data, 1, 0.2, dx=dx, plotflag=False,
                            precondflag=False)
            u1s = TVRegDiff(data, 1, 0.2, dx=dx, plotflag=False,
                            diffkernel='sq')

            # Loose tolerance for this
            self.assertLess(rms(u1a-targ), 0.3,
                            '[function: {0}]'.format(fname))
            self.assertLess(rms(u1s-targ), 0.3,
                            '[function: {0}]'.format(fname))

            # Tigher for more iterations
            u10a = TVRegDiff(data, 10, 0.2, dx=dx, plotflag=False,
                             precondflag=False)

            self.assertLess(rms(u10a-targ), 0.2,
                            '[function: {0}]'.format(fname))

            if plt:
                plt.title('scale = small')
                plt.plot(x, targ)
                plt.plot(x, u1a, label='u1a')
                plt.plot(x, u1s, label='u1s')
                plt.plot(x, u10a, label='u10a')
                plt.legend()
                plt.show()

    def test_large(self):

        n = 1000
        x = np.linspace(-5, 5, n)
        dx = x[1]-x[0]

        for fname, (f, df) in functions.items():

            data = f(x) + (np.random.random(n)-0.5)*0.05
            targ = df(x)

            u1a = TVRegDiff(data, 1, 0.2, dx=dx, plotflag=False,
                            scale='large')
            u1s = TVRegDiff(data, 1, 0.2, dx=dx, plotflag=False,
                            diffkernel='sq', scale='large')

            self.assertLess(rms(u1a-targ), 0.3,
                            '[function: {0}]'.format(fname))
            self.assertLess(rms(u1s-targ), 0.3,
                            '[function: {0}]'.format(fname))

            # Tigher for more iterations
            u10a = TVRegDiff(data, 10, 0.2, dx=dx, plotflag=False,
                             scale='large')

            self.assertLess(rms(u10a-targ), 0.2,
                            '[function: {0}]'.format(fname))

            if plt:
                plt.title('scale = large')
                plt.plot(x, targ)
                plt.plot(x, u1a, label='u1a')
                plt.plot(x, u1s, label='u1s')
                plt.plot(x, u10a, label='u10a')
                plt.legend()
                plt.show()


if __name__ == '__main__':
    unittest.main()
