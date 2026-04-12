"""Regression test: DHS hydraulic subproblem, Mat_Mul vs element-wise.

This test pins the behaviour of Solverz's matrix-vector calculus and
sparse-operation pipeline end-to-end on a non-trivial network (Barry
Island district heating case, 35 nodes / 35 pipes / 1 loop):

    ground truth  ← SolUtil's ``DhsFlow`` (pyomo/ipopt based)

    inline + elementwise  )  four paths through Solverz code generation
    inline + matmul       )  all expected to produce the same pipe mass
    module + elementwise  )  flow vector ``m``; any divergence is a bug
    module + matmul       )  in Mat_Mul / mutable-matrix Jacobian code

The Mat_Mul path exercises:

- precompute of ``V_ns @ m`` in the ``F_`` wrapper (constant-matrix
  Jacobian, hits the fast inline shortcut);
- ``Diag(K ⊙ |m|) + Diag(K ⊙ m ⊙ sign(m))`` contribution to the loop
  pressure Jacobian — a *mutable matrix* block the code generator must
  decompose into scatter-add loops.
"""
from __future__ import annotations

import copy
import os
import shutil
import sys

import numpy as np
import pandas as pd
import scipy.sparse.linalg

from Solverz import made_numerical, module_printer, nr_method

# Ensure the local heat_flow_mdl.py (next to this test) is importable
# whether pytest is run from the repo root or from within this directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat_flow_mdl import build_elementwise_model, build_matmul_model


def _load_ground_truth(datadir):
    """Run SolUtil's ``DhsFlow`` and return the ground-truth case + ``m``.

    The load factors in the workbook are applied first (mirroring the
    SolUtil test) so that the injection pattern matches what a real user
    would compute.
    """
    from SolUtil.energyflow.dhs_flow import DhsFlow

    xlsx = str(datadir / 'BarryIsland.xlsx')
    df = DhsFlow(xlsx)
    bench = pd.read_excel(xlsx, sheet_name=None, engine='openpyxl', index_col=None)
    df.phi *= np.asarray(bench['load_fac']['load_fac'])
    df.run(tee=False)
    assert df.run_succeed, "SolUtil DhsFlow did not converge on BarryIsland"

    # Signed node injection vector — computed directly from the ground
    # truth mass flow so that ``V @ m = m_inj`` is consistent.
    m_inj = df.A @ df.m
    return df, m_inj


def _solve(model, y0_seed=None, tol=1e-10):
    """Create a Solverz instance and solve via inline ``nr_method``."""
    spf, y0 = model.create_instance()
    if y0_seed is not None:
        y0.array[:] = y0_seed
    mdl = made_numerical(spf, y0, sparse=True)
    sol = nr_method(mdl, y0)
    assert sol.stats.succeed, "Inline Newton did not converge"
    return sol.y['m'], spf, y0


def _solve_module(model, name, directory, jit):
    """Render the model via ``module_printer`` and solve from ``y``."""
    spf, y0 = model.create_instance()
    printer = module_printer(spf, y0, name, directory=directory, jit=jit)
    printer.render()
    sys.path.insert(0, directory)
    import importlib
    mod = importlib.import_module(name)
    importlib.reload(mod)  # pick up any regenerated code on repeated runs
    mdl = mod.mdl
    y = mod.y
    sol = nr_method(mdl, y)
    assert sol.stats.succeed, f"Module Newton did not converge for {name}"
    return sol.y['m']


def test_heat_flow_hydraulic_matmul_regression(datadir):
    """Full regression: all four solver paths must agree with SolUtil."""
    df, m_inj = _load_ground_truth(datadir)
    hc = df.hc

    # A flat-but-non-zero initial guess so Newton has something to do.
    m_init = 0.5 * np.ones(hc['n_pipe'])

    # -------- 1. Element-wise, inline mode --------
    mdl_ew = build_elementwise_model(hc, m_init, m_inj)
    m_ew_inline, _, _ = _solve(mdl_ew)
    np.testing.assert_allclose(
        m_ew_inline, df.m, rtol=1e-6, atol=1e-8,
        err_msg='element-wise inline disagrees with SolUtil ground truth')

    # -------- 2. Mat_Mul, inline mode --------
    mdl_mm = build_matmul_model(hc, m_init, m_inj)
    m_mm_inline, spf_mm, y0_mm = _solve(mdl_mm)
    np.testing.assert_allclose(
        m_mm_inline, df.m, rtol=1e-6, atol=1e-8,
        err_msg='Mat_Mul inline disagrees with SolUtil ground truth')
    # Mat_Mul and element-wise inline must agree to machine precision —
    # they solve the same system and start from the same y0.
    np.testing.assert_allclose(
        m_mm_inline, m_ew_inline, rtol=1e-10, atol=1e-12,
        err_msg='Mat_Mul inline diverges from element-wise inline')

    # -------- 3. Element-wise, module printer (jit=True) --------
    tmp_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '_hf_generated')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    mdl_ew_mod = build_elementwise_model(hc, m_init, m_inj)
    m_ew_mod = _solve_module(mdl_ew_mod, 'heat_flow_ew_mod', tmp_dir, jit=True)
    np.testing.assert_allclose(
        m_ew_mod, df.m, rtol=1e-6, atol=1e-8,
        err_msg='element-wise module printer disagrees with SolUtil')

    # -------- 4. Mat_Mul, module printer (jit=True) --------
    mdl_mm_mod = build_matmul_model(hc, m_init, m_inj)
    m_mm_mod = _solve_module(mdl_mm_mod, 'heat_flow_mm_mod', tmp_dir, jit=True)
    np.testing.assert_allclose(
        m_mm_mod, df.m, rtol=1e-6, atol=1e-8,
        err_msg='Mat_Mul module printer disagrees with SolUtil')
    np.testing.assert_allclose(
        m_mm_mod, m_mm_inline, rtol=1e-10, atol=1e-12,
        err_msg='Mat_Mul module printer diverges from its inline twin')

    shutil.rmtree(tmp_dir)


def test_heat_flow_stepwise_jacobian_match(datadir):
    """Every Newton step must produce the same Jacobian in inline and
    module-printer Mat_Mul modes — this is the strict check that catches
    the class of bug we hit in v0.8.0 where the module printer froze its
    mutable-matrix Jacobian blocks at the initial values.
    """
    df, m_inj = _load_ground_truth(datadir)
    hc = df.hc
    m_init = 0.5 * np.ones(hc['n_pipe'])

    # Inline model
    mdl_mm = build_matmul_model(hc, m_init, m_inj)
    spf, y0 = mdl_mm.create_instance()
    mdl_inline = made_numerical(spf, y0, sparse=True)

    # Module printer (no jit so the comparison is pure Python arithmetic)
    tmp_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '_hf_step_generated')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    printer = module_printer(spf, y0, 'heat_flow_step', directory=tmp_dir, jit=False)
    printer.render()
    sys.path.insert(0, tmp_dir)
    import importlib
    if 'heat_flow_step' in sys.modules:
        del sys.modules['heat_flow_step']
    mod = importlib.import_module('heat_flow_step')
    mdl_mod = mod.mdl
    y_mod = mod.y

    # Start from a non-trivial guess so every pipe has a non-zero, non-
    # matching sign pattern — this forces the mutable matrix Jacobian to
    # be re-evaluated at each step.
    rng = np.random.default_rng(20260412)
    y_seed = rng.uniform(-2.0, 2.0, size=hc['n_pipe'])
    y_test = copy.deepcopy(y0)
    y_test.array[:] = y_seed
    y_test_mod = copy.deepcopy(y_mod)
    y_test_mod.array[:] = y_seed

    for step in range(6):
        F_i = mdl_inline.F(y_test, mdl_inline.p)
        F_m = mdl_mod.F(y_test_mod, mdl_mod.p)
        J_i = mdl_inline.J(y_test, mdl_inline.p)
        J_m = mdl_mod.J(y_test_mod, mdl_mod.p)
        np.testing.assert_allclose(
            F_i, F_m, rtol=1e-10, atol=1e-12,
            err_msg=f'F mismatch at step {step}')
        np.testing.assert_allclose(
            J_i.toarray(), J_m.toarray(), rtol=1e-10, atol=1e-12,
            err_msg=f'J mismatch at step {step} — '
                    f'max diff {abs(J_i - J_m).max():.2e}')
        # Drive the iteration forward using the inline J so both models
        # see the same next state.
        dy = scipy.sparse.linalg.spsolve(J_i, -F_i)
        y_test.array[:] = y_test.array + dy
        y_test_mod.array[:] = y_test_mod.array + dy

    shutil.rmtree(tmp_dir)


# ---- Regression tests for review findings ----

def _square_hc_and_inj(datadir):
    """Load the BarryIsland case and return ``(hc, m_inj, df)``."""
    from SolUtil.energyflow.dhs_flow import DhsFlow
    xlsx = str(datadir / 'BarryIsland.xlsx')
    df = DhsFlow(xlsx)
    bench = pd.read_excel(xlsx, sheet_name=None, engine='openpyxl', index_col=None)
    df.phi *= np.asarray(bench['load_fac']['load_fac'])
    df.run(tee=False)
    m_inj = df.A @ df.m
    return df, m_inj


def test_heat_flow_radial_no_loop_equation():
    """Finding 7: a radial (tree) network has no loops and therefore
    no loop-pressure equation. The builder must not synthesise an
    all-zero row or an empty ``Mat_Mul(L, ...)`` — it must emit only
    the mass-continuity equations and leave the system square
    (``n_pipe = n_node - 1`` for a tree).
    """
    import networkx as nx
    # 4-node chain: 0 → 1 → 2 → 3 (3 pipes, no loops).
    G = nx.DiGraph()
    G.add_edges_from([
        (0, 1, {'idx': 0}),
        (1, 2, {'idx': 1}),
        (2, 3, {'idx': 2}),
    ])
    n_node, n_pipe = 4, 3
    hc = {
        'n_node': n_node,
        'n_pipe': n_pipe,
        'G': G,
        'K': np.ones(n_pipe),
        # All-zero pinloop — radial network signal.
        'pinloop': np.zeros(n_pipe, dtype=float),
        'slack_node': np.array([0], dtype=int),
    }
    m_inj = np.array([-1.0, 0.0, 0.0, 1.0])
    m_init = np.full(n_pipe, 0.5)

    mdl_ew = build_elementwise_model(hc, m_init, m_inj)
    spf_ew, _ = mdl_ew.create_instance()
    assert not any(name.startswith('loop_pressure')
                   for name in spf_ew.EQNs.keys()), \
        f'radial elementwise model should have no loop equation; got ' \
        f'{list(spf_ew.EQNs.keys())}'
    # (n_node-1) balance rows = 3, matching n_pipe = 3 unknowns.
    assert spf_ew.eqn_size == n_pipe
    assert spf_ew.vsize == n_pipe

    mdl_mm = build_matmul_model(hc, m_init, m_inj)
    spf_mm, _ = mdl_mm.create_instance()
    assert not any(name.startswith('loop_pressure')
                   for name in spf_mm.EQNs.keys()), \
        f'radial Mat_Mul model should have no loop equation; got ' \
        f'{list(spf_mm.EQNs.keys())}'
    assert spf_mm.eqn_size == n_pipe
    assert spf_mm.vsize == n_pipe


def test_heat_flow_multi_slack_drops_one_row(datadir):
    """Finding 8: with multiple slack nodes the builder must still
    drop exactly **one** continuity row, not one row per slack. Dropping
    per-slack leaves the system underdetermined.
    """
    df, m_inj = _square_hc_and_inj(datadir)
    hc = dict(df.hc)
    # Fake multiple slack nodes — pretend 0 and 33 are both slacks.
    hc['slack_node'] = np.array([0, 33], dtype=hc['slack_node'].dtype)
    m_init = df.m.copy()

    mdl_ew = build_elementwise_model(hc, m_init, m_inj)
    spf_ew, _ = mdl_ew.create_instance()

    # With 35 nodes and 1 loop, dropping ONE row leaves
    # (n_node - 1) + 1 = n_node = 35 equations, matching n_pipe = 35.
    assert spf_ew.eqn_size == hc['n_pipe'], (
        f'multi-slack elementwise should drop exactly ONE continuity '
        f'row regardless of slack count; got {spf_ew.eqn_size} eqns '
        f'vs {hc["n_pipe"]} unknowns')

    mdl_mm = build_matmul_model(hc, m_init, m_inj)
    spf_mm, _ = mdl_mm.create_instance()
    assert spf_mm.eqn_size == hc['n_pipe'], (
        f'multi-slack Mat_Mul should drop exactly ONE continuity row; '
        f'got {spf_mm.eqn_size} vs {hc["n_pipe"]}')


def test_heat_flow_multi_loop_builder():
    """Finding 9: the elementwise builder must emit one scalar loop
    equation per row of a 2-D ``pinloop``. The current code incorrectly
    collapses multi-loop input into a single sum.
    """
    import networkx as nx
    # Tiny 4-node diamond with 5 pipes → 2 independent loops.
    #
    #         p0      p1
    #     0 -----> 1 -----> 3
    #     |                 ^
    #     | p2              | p3
    #     v                 |
    #     2 ----------------+
    #              p4
    G = nx.DiGraph()
    G.add_edges_from([
        (0, 1, {'idx': 0}),
        (1, 3, {'idx': 1}),
        (0, 2, {'idx': 2}),
        (2, 3, {'idx': 3}),
        (1, 2, {'idx': 4}),  # chord forming a second loop
    ])
    n_node, n_pipe = 4, 5
    K = np.ones(n_pipe)
    # Two loops, encoded as a 2-D (n_loop, n_pipe) array:
    #   loop 0:  +p0, +p1, -p3, -p2
    #   loop 1:  +p0, +p4, -p2
    pinloop = np.array([
        [1, 1, -1, -1, 0],
        [1, 0, -1, 0, 1],
    ], dtype=float)
    hc = {
        'n_node': n_node,
        'n_pipe': n_pipe,
        'G': G,
        'K': K,
        'pinloop': pinloop,
        'slack_node': np.array([0], dtype=int),
    }
    m_inj = np.array([-1.0, 0.0, 0.0, 1.0])

    mdl_ew = build_elementwise_model(hc, np.full(n_pipe, 0.5), m_inj)
    spf_ew, _ = mdl_ew.create_instance()
    loop_eqns = [name for name in spf_ew.EQNs
                 if name.startswith('loop_pressure')]
    assert len(loop_eqns) == 2, (
        f'multi-loop elementwise builder should emit one equation per '
        f'loop row; got {len(loop_eqns)}: {loop_eqns}')
    # System must be square: (n_node-1) mass balances + 2 loop eqns
    #                      = 3 + 2 = 5 = n_pipe
    assert spf_ew.eqn_size == n_pipe

    mdl_mm = build_matmul_model(hc, np.full(n_pipe, 0.5), m_inj)
    spf_mm, _ = mdl_mm.create_instance()
    # Mat_Mul builder emits ONE vector equation of length n_loop=2
    # (plus the mass_balance vector eqn of length n_node-1).
    assert spf_mm.eqn_size == n_pipe, (
        f'multi-loop Mat_Mul system not square: '
        f'{spf_mm.eqn_size} vs {n_pipe}')
