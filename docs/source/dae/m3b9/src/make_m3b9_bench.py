"""Write the m3b9 benchmarks that test_m3b9.py and test_m3b9_loopeqn.py compare against.

The benchmark is the scalar model integrated at rtol 1e-10, so it stands for
the exact trajectory: the tests integrate at rtol 1e-6 and their error is
measured against it. Run from this directory::

    python make_m3b9_bench.py
"""
from pathlib import Path

import numpy as np

from Solverz import Opt, Rodas, made_numerical
from test_m3b9 import TSPAN, build_m3b9

HERE = Path(__file__).resolve().parent

m3b9, y0 = build_m3b9(HERE / 'test_m3b9')
dae = made_numerical(m3b9, y0, sparse=True)
sol = Rodas(dae, TSPAN, y0, Opt(hinit=1e-5, rtol=1e-10, atol=1e-12))
print(f'{sol.stats.nstep} steps')

for datadir in ('test_m3b9', 'test_m3b9_loopeqn'):
    for name in ('delta', 'omega', 'Ux', 'Uy'):
        path = HERE / datadir / f'{name}_bench.npy'
        old = np.load(path)
        new = np.asarray(sol.Y[name])
        np.save(path, new)
        print(f'{datadir}/{name}_bench.npy: max |new - old| = {np.max(np.abs(new - old)):.3e}')
