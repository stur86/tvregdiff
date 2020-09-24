# tvregdiff

Python version of Rick Chartrand's algorithm for numerical differentiation of noisy data.
Requires Numpy and Scipy installed. Matplotlib optional for plotting.

Usage: 

```python
u = TVRegDiff(data, iter, alph, u0, scale, ep, dx, plotflag, diagflag, precondflag, diffkernel, cgtol, cgmaxit)
```

Test:

```bash
python tvregdiff.py test_data.dat
```
## New parameters

There are a few parameters added with respect to the original script, to allow for greater flexibility. These are:

* `precondflag`: if set to False, avoid using a preconditioner. Especially for `scale='small'` problems, sometimes the preconditioner can impede rather than help convergence, and it's useful to turn it off.
* `diffkernel`: by default is set to `'abs'`, which means the functional that will be optimised to find the derivative while keeping it smooth depends on the integral of |u'|. However it is also possible to set it to `'sq'`, which means using instead the integral of (u')^2. In the latter case, the derivative tends to come out smoother, and the need for using more than one iteration is much less. Try which one works best.
* `cgtol`: tolerance for the conjugate gradient optimisation, previously fixed.
* `cgmaxit` maximum number of iterations for the conjugate gradient optimisation, previously fixed.

## References

Rick Chartrand (rickc@lanl.gov), Apr. 10, 2011
Please cite Rick Chartrand, "Numerical differentiation of noisy, nonsmooth data," *ISRN Applied Mathematics*, Vol. 2011, Article ID 164564, 2011.

Algorithm adapted from the Matlab version found [here](https://sites.google.com/site/dnartrahckcir/home/tvdiff-code).


