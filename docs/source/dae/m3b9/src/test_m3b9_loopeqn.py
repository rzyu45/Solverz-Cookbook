"""m3b9 LoopEqn regression test for the TimeSeriesParam-in-J bug.

This is a LoopEqn-flavoured counterpart to :func:`test_m3b9` in
``test_m3b9.py``. Same machine model, same G66 short-circuit
profile, but the rectangular current-injection equations
``Ix_inj_i`` / ``Iy_inj_i`` are written as a single ``LoopEqn``
each (mirroring how ``SolMuseum.ae.eps_network.mdl(dyn=True,
loopeqn=True)`` emits them for production EPS models).

The shape exercised here is the one Solverz commit ``141ea3a``
silently broke: a LoopEqn body with a static ``Sum(Gbus[i,j]*Ux[j],
j)`` term PLUS a ``TimeSeriesParam[i] * Var[i]`` diagonal term.
Before the bug fix the ``is_constant_matrix_deri`` predicate
classified the entire ``-Gbus - Diag(G66)`` derivative as constant
(because every free symbol was a ``Para``) and baked the value at
the build-time TimeSeriesParam reading (``Gbus[6,6]`` plus zero).
F still updated correctly via ``get_v_t(t)``, but J froze — and
Rodas's modified-Newton iterations integrated the fault transient
with the un-faulted Jacobian, producing the silent
"Step rejected over 100 times" abort at the fault inception that
this test guards against.

The asserted-against benchmarks are the scalar ``test_m3b9``
trajectories. The LoopEqn and scalar formulations are
algebraically identical, so the integrated trajectories must
agree to the solver-tolerance noise floor.
"""
import numpy as np
import pandas as pd

from Solverz import (
    Eqn,
    Idx,
    LoopEqn,
    Model,
    Ode,
    Opt,
    Param,
    Rodas,
    Sum,
    TimeSeriesParam,
    Var,
    cos,
    sin,
    module_printer,
)
from scipy.sparse import csc_array

import importlib
import sys
import tempfile
import uuid


def _mdl_from_module(spf, y0, jit: bool = True):
    """Render via ``module_printer`` and import. LoopEqn cannot run
    through the inline ``made_numerical`` path (issue #132)."""
    mod_name = f"_sz_m3b9_loop_{uuid.uuid4().hex[:8]}"
    d = tempfile.mkdtemp()
    printer = module_printer(spf, y0, mod_name, directory=d, jit=jit)
    printer.render()
    sys.path.insert(0, d)
    mod = importlib.import_module(mod_name)
    return mod.mdl, mod.y


def test_m3b9_loopeqn(datadir):
    """LoopEqn variant of test_m3b9. Same physics, same G66 fault,
    must produce the same Ux/Uy/delta/omega trajectories."""
    m = Model()
    m.omega = Var('omega', [1, 1, 1])
    m.delta = Var('delta', [0.0625815077879868, 1.06638275203221, 0.944865048677501])
    m.Ux = Var('Ux', [1.04000110267534, 1.01157932564567, 1.02160343921907,
                      1.02502063033405, 0.993215117729926, 1.01056073782038,
                      1.02360471178264, 1.01579907336413, 1.03174403980626])
    m.Uy = Var('Uy', [9.38510394478286e-07, 0.165293826097057, 0.0833635520284917,
                      -0.0396760163416718, -0.0692587531054159, -0.0651191654677445,
                      0.0665507083524658, 0.0129050646926083, 0.0354351211556429])
    m.Ixg = Var('Ixg', [0.688836021737262, 1.57988988391346, 0.817891311823357])
    m.Iyg = Var('Iyg', [-0.260077644814056, 0.192406178191528, 0.173047791590276])
    # Pad Ixg/Iyg to length 9 so we can index into them by bus number
    # in the LoopEqn body. Buses 0..2 carry the generator currents;
    # buses 3..8 are pinned to zero via the scalar Eqns below
    # (matching test_m3b9.py's ``rhs1 = 0 if i >= 3``).
    m.Ix = Var('Ix', list(m.Ixg.value) + [0.0] * 6)
    m.Iy = Var('Iy', list(m.Iyg.value) + [0.0] * 6)
    m.Pm = Param('Pm', [0.7164, 1.6300, 0.8500])
    m.D = Param('D', [10, 10, 10])
    m.Tj = Param('Tj', [47.2800, 12.8000, 6.0200])
    m.ra = Param('ra', [0.0000, 0.0000, 0.0000])
    wb = 376.991118430775
    m.Edp = Param('Edp', [0.0000, 0.0000, 0.0000])
    m.Eqp = Param('Eqp', [1.05636632091501, 0.788156757672709, 0.767859471854610])
    m.Xdp = Param('Xdp', [0.0608, 0.1198, 0.1813])
    m.Xqp = Param('Xqp', [0.0969, 0.8645, 1.2578])

    Pe = m.Ux[0:3] * m.Ixg + m.Uy[0:3] * m.Iyg + (m.Ixg ** 2 + m.Iyg ** 2) * m.ra
    m.rotator_eqn = Ode(name='rotator speed',
                        f=(m.Pm - Pe - m.D * (m.omega - 1)) / m.Tj,
                        diff_var=m.omega)
    omega_coi = (m.Tj[0] * m.omega[0] + m.Tj[1] * m.omega[1] + m.Tj[2] * m.omega[2]) / (
            m.Tj[0] + m.Tj[1] + m.Tj[2])
    m.delta_eq = Ode(name='Delta equation',
                     f=wb * (m.omega - omega_coi),
                     diff_var=m.delta)
    m.Ed_prime = Eqn(name='Ed_prime',
                     eqn=(m.Edp - sin(m.delta) * (m.Ux[0:3] + m.ra * m.Ixg - m.Xqp * m.Iyg)
                          + cos(m.delta) * (m.Uy[0:3] + m.ra * m.Iyg + m.Xqp * m.Ixg)))
    m.Eq_prime = Eqn(name='Eq_prime',
                     eqn=(m.Eqp - cos(m.delta) * (m.Ux[0:3] + m.ra * m.Ixg - m.Xdp * m.Iyg)
                          - sin(m.delta) * (m.Uy[0:3] + m.ra * m.Iyg + m.Xdp * m.Ixg)))
    df = pd.read_excel(datadir / 'test_m3b9.xlsx',
                       sheet_name=None,
                       engine='openpyxl',
                       header=None)
    G_full = np.asarray(df['G'])
    B_full = np.asarray(df['B'])

    # Insert G66 as a TimeSeriesParam — exactly the spec from
    # test_m3b9.py. We achieve this by overlaying a scalar diagonal
    # vector ``G_shunt`` onto the (i,i) diagonal of an otherwise
    # static Gbus: ``Gbus[i,j]`` carries the full network admittance
    # MINUS the original G[6,6] entry at (6,6); ``G_shunt[6]``
    # carries the time-varying replacement value G66(t). The
    # effective diagonal at runtime is then
    #   Gbus[6,6] + G_shunt[6] = 0 + G66(t) = G66(t),
    # matching the scalar test's getGitem dispatch.
    G_static = G_full.copy()
    G_static[6, 6] = 0.0  # the (6,6) entry comes entirely from G_shunt below
    m.Gbus = Param('Gbus', csc_array(G_static), dim=2, sparse=True)
    m.Bbus = Param('Bbus', csc_array(B_full), dim=2, sparse=True)
    nb = 9
    G_shunt_value = np.zeros(nb)
    G_shunt_value[6] = G_full[6, 6]  # pre-fault diagonal initial value
    m.G_shunt = TimeSeriesParam(
        'G_shunt',
        v_series=[G_full[6, 6], 10000.0, 10000.0,
                  G_full[6, 6], G_full[6, 6]],
        time_series=[0.0, 0.002, 0.03, 0.032, 10.0],
        index=6,
        value=G_shunt_value,
    )

    # Use unique Idx names — sympy keeps Idx globally by name and
    # other LoopEqn tests in the suite would otherwise pollute the
    # substitution state.
    suffix = uuid.uuid4().hex[:8]
    i = Idx(f'_m3b9_i_{suffix}', nb)
    j = Idx(f'_m3b9_j_{suffix}', nb)

    # The LoopEqn body is the bug shape: ``Sum(Gbus[i,j]*Ux[j], j)``
    # (static) plus ``G_shunt[i] * Ux[i]`` (TimeSeriesParam). The
    # ``Ix[i]`` term carries the generator-current contribution on
    # buses 0..2 and is pinned to zero on buses 3..8 by the scalar
    # pin Eqns below.
    body_ix = (m.Ix[i]
               - Sum(m.Gbus[i, j] * m.Ux[j], j)
               + Sum(m.Bbus[i, j] * m.Uy[j], j)
               - m.G_shunt[i] * m.Ux[i])
    body_iy = (m.Iy[i]
               - Sum(m.Gbus[i, j] * m.Uy[j], j)
               - Sum(m.Bbus[i, j] * m.Ux[j], j)
               - m.G_shunt[i] * m.Uy[i])
    m.Ix_inj = LoopEqn('Ix_inj', outer_index=i, body=body_ix, model=m)
    m.Iy_inj = LoopEqn('Iy_inj', outer_index=i, body=body_iy, model=m)

    # Pin Ix/Iy to Ixg/Iyg on the generator buses (0..2) and to
    # zero on the load/transit buses (3..8). The LoopEqn already
    # enforced ix_inj_i = 0 for every i; these pins fix the
    # remaining ``Ix[i]`` / ``Iy[i]`` degrees of freedom.
    for k in range(3):
        m.add(Eqn(f'Ix_pin_gen_{k}', m.Ix[k] - m.Ixg[k]))
        m.add(Eqn(f'Iy_pin_gen_{k}', m.Iy[k] - m.Iyg[k]))
    for k in range(3, 9):
        m.add(Eqn(f'Ix_pin_zero_{k}', m.Ix[k]))
        m.add(Eqn(f'Iy_pin_zero_{k}', m.Iy[k]))

    m3b9, y0 = m.create_instance()
    mdl, y = _mdl_from_module(m3b9, y0, jit=True)

    sol = Rodas(mdl, np.linspace(0, 10, 1001), y, Opt(hinit=1e-5))

    # Compare against the scalar-test benchmarks. The two
    # formulations are algebraically identical (same physics, same
    # equations rearranged into LoopEqn templates), so the
    # trajectories must agree to scalar-test tolerances.
    with open(datadir / 'delta_bench.npy', 'rb') as f:
        delta_bench = np.load(f)
    np.testing.assert_allclose(sol.Y['delta'], delta_bench, rtol=1e-4, atol=1e-5)

    with open(datadir / 'omega_bench.npy', 'rb') as f:
        omega_bench = np.load(f)
    np.testing.assert_allclose(sol.Y['omega'], omega_bench, rtol=1e-4, atol=1e-5)

    with open(datadir / 'Ux_bench.npy', 'rb') as f:
        Ux_bench = np.load(f)
    np.testing.assert_allclose(sol.Y['Ux'], Ux_bench, rtol=1e-4, atol=1e-5)

    with open(datadir / 'Uy_bench.npy', 'rb') as f:
        Uy_bench = np.load(f)
    np.testing.assert_allclose(sol.Y['Uy'], Uy_bench, rtol=1e-2, atol=1e-3)
