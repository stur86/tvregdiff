# tvregdiff

Python version of Rick Chartrand's algorithm for numerical differentiation of noisy data.
Requires Numpy and Scipy installed. Matplotlib optional for plotting.

Usage: 

```python
u = TVRegDiff(data, iter, alph, u0, scale, ep, dx, plotflag, diagflag)
```

Test:

```bash
python tvregdiff.py test_data.dat
```

Rick Chartrand (rickc@lanl.gov), Apr. 10, 2011
Please cite Rick Chartrand, "Numerical differentiation of noisy, nonsmooth data," *ISRN Applied Mathematics*, Vol. 2011, Article ID 164564, 2011.

Algorithm adapted from the Matlab version found [here](https://sites.google.com/site/dnartrahckcir/home/tvdiff-code).


