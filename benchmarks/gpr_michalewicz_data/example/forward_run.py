"""Evaluate the 10-D Michalewicz function for pestpp-sm.

This implements the same function as laGPy's MIC.py, which calls
``benchmark_functions.Michalewicz(n_dimensions=10)``:

    f(x) = -sum_i sin(x_i) * sin^{2m}((i+1) * x_i^2 / pi),   m = 10

Parameter values are read from ``par.dat`` (columns x1..x10) and the result is
written to ``obs.dat`` as observation ``func``.
"""

import math

M = 10
PARS = ["x{0}".format(i) for i in range(1, 11)]


def michalewicz(point, m=M):
    """10-D Michalewicz (benchmark_functions.Michalewicz, m=10)."""
    s = 0.0
    for i, xi in enumerate(point):
        s += math.sin(xi) * math.sin((i + 1) * xi * xi / math.pi) ** (2 * m)
    return -s


pars = {}
with open("par.dat", "r") as f:
    for line in f:
        t = line.split()
        if len(t) >= 2:
            pars[t[0].strip()] = float(t[1])

point = [pars[p] for p in PARS]
val = michalewicz(point)

with open("obs.dat", "w") as f:
    f.write("func {0:24.16E}\n".format(val))
