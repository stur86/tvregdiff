#!/usr/bin/env python
"""
Python function to estimate derivatives from noisy data based on
Rick Chartrand's Total Variation Regularized Numerical 
Differentiation (TVDiff) algorithm.

Example:
>>> u = TVRegDiff(data, iter, alph, u0, scale, ep, dx,  
...               plotflag, diagflag)    

Rick Chartrand (rickc@lanl.gov), Apr. 10, 2011
Please cite Rick Chartrand, "Numerical differentiation of noisy,
nonsmooth data," ISRN Applied Mathematics, Vol. 2011, Article ID 164564,
2011.

Copyright notice:
Copyright 2010. Los Alamos National Security, LLC. This material
was produced under U.S. Government contract DE-AC52-06NA25396 for
Los Alamos National Laboratory, which is operated by Los Alamos
National Security, LLC, for the U.S. Department of Energy. The
Government is granted for, itself and others acting on its
behalf, a paid-up, nonexclusive, irrevocable worldwide license in
this material to reproduce, prepare derivative works, and perform
publicly and display publicly. Beginning five (5) years after
(March 31, 2011) permission to assert copyright was obtained,
subject to additional five-year worldwide renewals, the
Government is granted for itself and others acting on its behalf
a paid-up, nonexclusive, irrevocable worldwide license in this
material to reproduce, prepare derivative works, distribute
copies to the public, perform publicly and display publicly, and
to permit others to do so. NEITHER THE UNITED STATES NOR THE
UNITED STATES DEPARTMENT OF ENERGY, NOR LOS ALAMOS NATIONAL
SECURITY, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY,
EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF
ANY INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR
REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
RIGHTS.

BSD License notice:
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

     Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     disclaimer.
     Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials
     provided with the distribution.
     Neither the name of Los Alamos National Security nor the names of its
     contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

#########################################################
#                                                       #
# Python translation by Simone Sturniolo                #
# Rutherford Appleton Laboratory, STFC, UK (2014)       #
# simonesturniolo@gmail.com                             #
#                                                       #
#########################################################
"""

import sys

import logging
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg as splin

_has_matplotlib = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    _has_matplotlib = False
    logging.warning("Matplotlib is not installed - plotting "
                    "functionality disabled")


def log_iteration(ii, s0, u, g):
    relative_change = np.linalg.norm(s0) / np.linalg.norm(u)
    g_norm = np.linalg.norm(g)
    logging.info('iteration {0:4d}: relative change = {1:.3e}, '
                 'gradient norm = {2:.3e}\n'.format(ii,
                                                    relative_change,
                                                    g_norm))


def TVRegDiff(data, itern, alph, u0=None, scale='small', ep=1e-6, dx=None,
              plotflag=_has_matplotlib, diagflag=True):
    """
    Estimate derivatives from noisy data based using the Total 
    Variation Regularized Numerical Differentiation (TVDiff) 
    algorithm.

    Parameters
    ----------
    data : ndarray
        One-dimensional array containing series data to be
        differentiated.
    itern : int
        Number of iterations to run the main loop.  A stopping
        condition based on the norm of the gradient vector g
        below would be an easy modification.  No default value.
    alph : float    
        Regularization parameter.  This is the main parameter
        to fiddle with.  Start by varying by orders of
        magnitude until reasonable results are obtained.  A
        value to the nearest power of 10 is usally adequate.
        No default value.  Higher values increase
        regularization strenght and improve conditioning.
    u0 : ndarray, optional
        Initialization of the iteration.  Default value is the
        naive derivative (without scaling), of appropriate
        length (this being different for the two methods).
        Although the solution is theoretically independent of
        the initialization, a poor choice can exacerbate
        conditioning issues when the linear system is solved.
    scale : {large' or 'small' (case insensitive)}, str, optional   
        Default is 'small'.  'small' has somewhat better boundary
        behavior, but becomes unwieldly for data larger than
        1000 entries or so.  'large' has simpler numerics but
        is more efficient for large-scale problems.  'large' is
        more readily modified for higher-order derivatives,
        since the implicit differentiation matrix is square.
    ep : float, optional 
        Parameter for avoiding division by zero.  Default value
        is 1e-6.  Results should not be very sensitive to the
        value.  Larger values improve conditioning and
        therefore speed, while smaller values give more
        accurate results with sharper jumps.
    dx : float, optional    
        Grid spacing, used in the definition of the derivative
        operators.  Default is the reciprocal of the data size.
    plotflag : bool, optional
        Flag whether to display plot at each iteration.
        Default is True.  Useful, but adds significant
        running time.
    diagflag : bool, optional
        Flag whether to display diagnostics at each
        iteration.  Default is True.  Useful for diagnosing
        preconditioning problems.  When tolerance is not met,
        an early iterate being best is more worrying than a
        large relative residual.

    Returns
    -------
    u : ndarray
        Estimate of the regularized derivative of data.  Due to
        different grid assumptions, length(u) = length(data) + 1
        if scale = 'small', otherwise length(u) = length(data).
    """

    # Make sure we have a column vector
    data = np.array(data)
    assert len(data.shape) == 1, "data is not one-dimensional"
    # Get the data size.
    n = len(data)

    # Default checking. (u0 is done separately within each method.)
    if dx is None:
        dx = 1.0 / n

    # Different methods for small- and large-scale problems.
    if (scale.lower() == 'small'):

        # Construct differentiation matrix.
        c = np.ones(n + 1) / dx
        D = sparse.spdiags([-c, c], [0, 1], n, n + 1)

        DT = D.transpose()

        # Construct antidifferentiation operator and its adjoint.
        def A(x): return (np.cumsum(x) - 0.5 * (x + x[0]))[1:] * dx

        def AT(w): return (sum(w) * np.ones(n + 1) -
                           np.transpose(np.concatenate(([sum(w) / 2.0],
                                                        np.cumsum(w) -
                                                        w / 2.0)))) * dx

        # Default initialization is naive derivative

        if u0 is None:
            u0 = np.concatenate(([0], np.diff(data), [0]))

        u = u0
        # Since Au( 0 ) = 0, we need to adjust.
        ofst = data[0]
        # Precompute.
        ATb = AT(ofst - data)        # input: size n

        # Main loop.
        for ii in range(1, itern+1):
            # Diagonal matrix of weights, for linearizing E-L equation.
            Q = sparse.spdiags(1. / (np.sqrt((D * u)**2 + ep)), 0, n, n)
            # Linearized diffusion matrix, also approximation of Hessian.
            L = dx * DT * Q * D

            # Gradient of functional.
            g = AT(A(u)) + ATb + alph * L * u

            # Prepare to solve linear equation.
            tol = 1e-4
            maxit = 100
            # Simple preconditioner.
            P = alph * sparse.spdiags(L.diagonal() + 1, 0, n + 1, n + 1)

            def linop(v): return (alph * L * v + AT(A(v)))
            linop = splin.LinearOperator((n + 1, n + 1), linop)

            if diagflag:
                [s, info_i] = sparse.linalg.cg(
                    linop, g, x0=None, tol=tol, maxiter=maxit, callback=None,
                    M=P, atol='legacy')
                log_iteration(ii, s[0], u, g)
                if (info_i > 0):
                    logging.warning(
                        "WARNING - convergence to tolerance not achieved!")
                elif (info_i < 0):
                    logging.warning("WARNING - illegal input or breakdown")
            else:
                [s, info_i] = sparse.linalg.cg(
                    linop, g, x0=None, tol=tol, maxiter=maxit, callback=None,
                    M=P, atol='legacy')
            # Update solution.
            u = u - s
            # Display plot.
            if plotflag:
                plt.plot(u)
                plt.show()

    elif (scale.lower() == 'large'):

        # Construct anti-differentiation operator and its adjoint.
        def A(v): return np.cumsum(v)

        def AT(w): return (sum(w) * np.ones(len(w)) -
                           np.transpose(np.concatenate(([0.0],
                                                        np.cumsum(w[:-1])))))
        # Construct differentiation matrix.
        c = np.ones(n)
        D = sparse.spdiags([-c, c], [0, 1], n, n) / dx
        mask = np.ones((n, n))
        mask[-1, -1] = 0.0
        D = sparse.dia_matrix(D.multiply(mask))
        DT = D.transpose()
        # Since Au( 0 ) = 0, we need to adjust.
        data = data - data[0]
        # Default initialization is naive derivative.
        if u0 is None:
            u0 = np.concatenate(([0], np.diff(data)))
        u = u0
        # Precompute.
        ATd = AT(data)

        # Main loop.
        for ii in range(1, itern + 1):
            # Diagonal matrix of weights, for linearizing E-L equation.
            Q = sparse.spdiags(1. / np.sqrt((D*u)**2.0 + ep), 0, n, n)
            # Linearized diffusion matrix, also approximation of Hessian.
            L = DT * Q * D
            # Gradient of functional.
            g = AT(A(u)) - ATd
            g = g + alph * L * u
            # Build preconditioner.
            c = np.cumsum(range(n, 0, -1))
            B = alph * L + sparse.spdiags(c[::-1], 0, n, n)
            # droptol = 1.0e-2
            R = sparse.dia_matrix(np.linalg.cholesky(B.todense()))
            # Prepare to solve linear equation.
            tol = 1.0e-4
            maxit = 100

            def linop(v): return (alph * L * v + AT(A(v)))
            linop = splin.LinearOperator((n, n), linop)

            if diagflag:
                [s, info_i] = sparse.linalg.cg(
                    linop, -g, x0=None, tol=tol, maxiter=maxit, callback=None,
                    M=np.dot(R.transpose(), R), atol='legacy')
                log_iteration(ii, s[0], u, g)
                if (info_i > 0):
                    logging.warning(
                        "WARNING - convergence to tolerance not achieved!")
                elif (info_i < 0):
                    logging.warning("WARNING - illegal input or breakdown")

            else:
                [s, info_i] = sparse.linalg.cg(
                    linop, -g, x0=None, tol=tol, maxiter=maxit, callback=None,
                    M = np.dot(R.transpose(), R), atol='legacy')
            # Update current solution
            u = u + s
            # Display plot
            if plotflag:
                plt.plot(u / dx)
                plt.show()

        u = u / dx

    return u


if __name__ == "__main__":

    # Command line operation
    import argparse as ap

    parser = ap.ArgumentParser(description='Compute the derivative of a '
                               'noisy function with the TVRegDiff method')
    parser.add_argument('dat_file', type=str,
                        help='Tabulated ASCII file with the data'
                        ' to differentiate')
    parser.add_argument('-iter', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('-colx', type=int, default=0,
                        help='Index of the column containing the X data '
                        '(must be regularly spaced)')
    parser.add_argument('-coly', type=int, default=1,
                        help='Index of the column containing the Y data')
    parser.add_argument('-a', type=float, default=5e-2,
                        help='Regularization parameter')
    parser.add_argument('-ep', type=float, default=1e-5,
                        help='Parameter for avoiding division by zero')
    parser.add_argument('-lscale', action='store_true', default=False,
                        help='Use Large instead of Small scale algorithm')
    parser.add_argument('-plot', action='store_true', default=False,
                        help='Plot result with Matplotlib at the end')

    args = parser.parse_args()

    data = np.loadtxt(args.dat_file)
    X = data[:, args.colx]
    Y = data[:, args.coly]

    dX = X[1] - X[0]

    dYdX = TVRegDiff(Y, args.iter, args.a, dx=dX, ep=args.ep,
                     scale=('large' if args.lscale else 'small'), plotflag=False)

    for x, y in zip(X, dYdX):
        print(x, y)

    if (_has_matplotlib):

        Xext = np.concatenate([X - dX/2.0, [X[-1] + dX/2]])

        plt.plot(X, Y, label='f(x)', c=(0.2, 0.2, 0.2), lw=0.5)
        plt.plot(X, np.gradient(Y, dX), label='df/dx (numpy)', c=(0, 0.3, 0.8), lw=1)
        plt.plot(Xext, dYdX, label='df/dx (TVRegDiff)', c=(0.8, 0.3, 0.0), lw=1)
        plt.legend()
        plt.show()
