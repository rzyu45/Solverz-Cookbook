"""Benchmark eps_network's loopeqn=False vs loopeqn=True paths on case30.

Phases measured (each reported separately):

  1. Model construction      — ``eps_network(pf).mdl(dyn=False, loopeqn=...)``
  2. create_instance         — ``m.create_instance()``
  3. Module render (jit=True) — ``module_printer(...).render()``
  4. Numba JIT first-call    — fresh subprocess: ``import mod; mdl.F(); mdl.J()``
  5. @njit function count    — grep ``^def inner_`` in num_func.py
  6. Hot F / J (us, 500 avg) — steady-state per-call time (subprocess)
  7. NR end-to-end (s)        — ``nr_method(mdl, y)`` (subprocess)

case30 data lives in ``test_pf_jac/`` (pf.mat + pq.mat). Since
``PowerFlow`` requires an .xlsx mpc file, this script builds a
``MockPF`` shim populated from the .mat files that exposes the
attributes ``eps_network`` reads (Vm, Va, Pg/Qg/Pd/Qd, nb, Ybus, Gbus,
Bbus, idx_slack/idx_pv/idx_pq) plus a no-op ``.run()``.

Prints a summary table at the end.
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

OUT_ROOT = tempfile.mkdtemp(prefix="bench_loopeqn_pf_")
CASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_pf_jac')


def _load_mock_pf():
    """Load case30 .mat data and wrap it in a PowerFlow-like shim.

    ``eps_network`` calls ``pf.run()`` in its constructor and then
    accesses ``Vm/Va/Pg/Qg/Pd/Qd/nb/Ybus/idx_slack/idx_pv/idx_pq``.
    Since case30's pf.mat already gives us a PRE-converged voltage
    solution (V is the solved bus voltage), we don't need to actually
    solve a Newton iteration here — ``pf.run()`` becomes a no-op.
    """
    import numpy as np
    from scipy.io import loadmat
    from scipy.sparse import csc_array

    sys_data = loadmat(os.path.join(CASE_DATA_DIR, 'pf.mat'))
    PQ = loadmat(os.path.join(CASE_DATA_DIR, 'pq.mat'))

    V = sys_data['V'].reshape(-1)
    nb = V.shape[0]
    Ybus = sys_data['Ybus']
    if not hasattr(Ybus, 'tocsc'):
        Ybus = csc_array(Ybus)
    else:
        Ybus = Ybus.tocsc()

    ref = (sys_data['ref'] - 1).reshape(-1)
    pv = (sys_data['pv'] - 1).reshape(-1)
    pq = (sys_data['pq'] - 1).reshape(-1)

    mbase = 100.0
    Pg = PQ['Pg'].reshape(-1) / mbase
    Qg = PQ['Qg'].reshape(-1) / mbase
    Pd = PQ['Pd'].reshape(-1) / mbase
    Qd = PQ['Qd'].reshape(-1) / mbase

    class MockPF:
        pass

    pf = MockPF()
    pf.Vm = np.abs(V).astype(float).copy()
    pf.Va = np.angle(V).astype(float).copy()
    pf.Pg = Pg.astype(float).copy()
    pf.Qg = Qg.astype(float).copy()
    pf.Pd = Pd.astype(float).copy()
    pf.Qd = Qd.astype(float).copy()
    pf.nb = nb
    pf.Ybus = Ybus
    pf.Gbus = Ybus.real
    pf.Bbus = Ybus.imag
    pf.idx_slack = np.asarray(ref, dtype=int)
    pf.idx_pv = np.asarray(pv, dtype=int)
    pf.idx_pq = np.asarray(pq, dtype=int)
    pf.baseMVA = mbase
    pf.run = lambda: None  # no-op; pf.mat data is already solved
    return pf


# --------------------------------------------------------------------
# Timing helpers.
# --------------------------------------------------------------------

def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, t1 - t0


def _clean_numba_cache_for_module(module_dir):
    """Delete __pycache__ + .nbi/.nbc so import triggers cold compile."""
    for root, dirs, files in os.walk(module_dir):
        for d in list(dirs):
            if d == '__pycache__':
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)
        for f in files:
            if f.endswith('.nbi') or f.endswith('.nbc'):
                try:
                    os.remove(os.path.join(root, f))
                except OSError:
                    pass


def _count_njit_inner_funcs(num_func_path):
    """Count ``^def inner_`` occurrences in num_func.py (each is @njit)."""
    if not os.path.exists(num_func_path):
        return None
    with open(num_func_path, 'r') as f:
        text = f.read()
    return len(re.findall(r'^def inner_', text, flags=re.MULTILINE))


# --------------------------------------------------------------------
# Phase runners.
# --------------------------------------------------------------------

def _add_pq_pins(m, pf, loopeqn):
    """Add pin equations to square the system for NR end-to-end.

    ``eps_network.mdl(dyn=False)`` declares ``Pg/Qg/Pd/Qd`` as ``Var``
    (driven by upstream components in a larger IES model). Standalone,
    the system is therefore under-determined. The legacy path writes
    P+Q balance for ALL buses (pv+pq+ref) so it is also over-
    determined relative to its non-flat ``Vm[pq] / Va[pv+pq]`` Vars
    once we pin every Pg/Qg/Pd/Qd. To make BOTH paths square in a
    benchmark-fair way we:

    * Pin ``Pg/Qg/Pd/Qd`` to their initial values (matching equation
      kind: LoopEqn for ``loopeqn=True``, per-index ``Eqn`` for
      ``loopeqn=False``).
    * For the legacy path, additionally pin the otherwise-
      unconstrained PG/QG entries that show up as redundant rows in
      the legacy P/Q expansion (P at slack, Q at slack+pv) by leaving
      Pg/Qg unpinned at those indices — this is impossible without
      surgery on the eps_network output, so instead we DROP the pins
      for those indices so the row count matches the column count.

    Returns the resulting count balance for the caller to decide
    whether NR can run.
    """
    import numpy as np
    from Solverz import Eqn, Param
    from Solverz import Idx, LoopEqn

    nb = pf.nb
    ref = pf.idx_slack.tolist()
    pv = pf.idx_pv.tolist()
    # Bus indices where each Var component should be pinned.
    # In the legacy path eps_network already adds 1 P-balance per ref
    # bus and 1 Q-balance per (ref+pv) bus that effectively over-
    # constrain Pg(ref) and Qg(ref+pv). To square the system we DROP
    # pins for those entries (they are determined by the P/Q balance
    # at slack/pv buses).
    if loopeqn:
        # LoopEqn path is more "rectangular": Vm/Va are full nb-sized
        # so we add 1 pin per Var component except those already
        # constrained by Vm_pin / Va_pin. ref+pv Vm and ref Va are
        # already pinned, so Pg(ref), Qg(ref+pv) must NOT be pinned
        # (they are the natural slack/PV unknowns).
        Pg_pin_idx = np.array([i for i in range(nb) if i not in ref], dtype=int)
        Qg_pin_idx = np.array([i for i in range(nb) if i not in ref + pv], dtype=int)
        Pd_pin_idx = np.arange(nb, dtype=int)
        Qd_pin_idx = np.arange(nb, dtype=int)

        m.Pg_init = Param('Pg_init', pf.Pg[Pg_pin_idx].copy())
        m.Qg_init = Param('Qg_init', pf.Qg[Qg_pin_idx].copy())
        m.Pd_init = Param('Pd_init', pf.Pd.copy())
        m.Qd_init = Param('Qd_init', pf.Qd.copy())

        m.Pg_pin_idx = Param('Pg_pin_idx', Pg_pin_idx, dtype=int)
        m.Qg_pin_idx = Param('Qg_pin_idx', Qg_pin_idx, dtype=int)

        i_pg = Idx('i_pg_pin', len(Pg_pin_idx))
        m.Pg_pin = LoopEqn('Pg_pin', outer_index=i_pg,
                           body=m.Pg[m.Pg_pin_idx[i_pg]] - m.Pg_init[i_pg],
                           model=m)
        i_qg = Idx('i_qg_pin', len(Qg_pin_idx))
        m.Qg_pin = LoopEqn('Qg_pin', outer_index=i_qg,
                           body=m.Qg[m.Qg_pin_idx[i_qg]] - m.Qg_init[i_qg],
                           model=m)
        i_pd = Idx('i_pd_pin', nb)
        m.Pd_pin = LoopEqn('Pd_pin', outer_index=i_pd,
                           body=m.Pd[i_pd] - m.Pd_init[i_pd], model=m)
        i_qd = Idx('i_qd_pin', nb)
        m.Qd_pin = LoopEqn('Qd_pin', outer_index=i_qd,
                           body=m.Qd[i_qd] - m.Qd_init[i_qd], model=m)
    else:
        # Per-index pins matching the legacy per-bus eqn style.
        # Same drop-rule: don't pin Pg(ref), Qg(ref+pv).
        Pg0 = pf.Pg.copy()
        Qg0 = pf.Qg.copy()
        Pd0 = pf.Pd.copy()
        Qd0 = pf.Qd.copy()
        for i in range(nb):
            if i not in ref:
                m.__dict__[f'Pg_pin_{i}'] = Eqn(f'Pg_pin_{i}',
                                                m.Pg[i] - float(Pg0[i]))
            if i not in ref + pv:
                m.__dict__[f'Qg_pin_{i}'] = Eqn(f'Qg_pin_{i}',
                                                m.Qg[i] - float(Qg0[i]))
            m.__dict__[f'Pd_pin_{i}'] = Eqn(f'Pd_pin_{i}',
                                            m.Pd[i] - float(Pd0[i]))
            m.__dict__[f'Qd_pin_{i}'] = Eqn(f'Qd_pin_{i}',
                                            m.Qd[i] - float(Qd0[i]))


def run_in_process_phases(label, loopeqn):
    """Phases 1-3 + njit-count: build model, create_instance, render."""
    from SolMuseum.ae import eps_network
    from Solverz import module_printer

    pf = _load_mock_pf()
    epsn = eps_network(pf)

    # Phase 1: model build (the eps_network mdl call itself is what we
    # want to time; the pq-pin additions below are common scaffolding
    # to make the system square for NR — those are NOT counted in
    # ``t_build`` so the comparison stays focused on eps_network).
    m, t_build = _timed(epsn.mdl, dyn=False, loopeqn=loopeqn)

    _add_pq_pins(m, pf, loopeqn)

    # Phase 2: create_instance
    (spf, y0), t_inst = _timed(m.create_instance)

    # Phase 3: module render with jit=True
    out_dir = os.path.join(OUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)
    mod_name = f'pf_{label}_mod'
    _, t_render = _timed(lambda: module_printer(
        spf, y0, mod_name, directory=out_dir, jit=True).render())

    # Count njit inner_ functions in the rendered num_func.py
    num_func_path = os.path.join(out_dir, mod_name, 'num_func.py')
    n_njit = _count_njit_inner_funcs(num_func_path)

    return {
        'build': t_build,
        'create_instance': t_inst,
        'render': t_render,
        'njit_count': n_njit,
        'module_dir': out_dir,
        'mod_name': mod_name,
        'n_eqn_families': len(spf.EQNs),
        'eqn_size': spf.eqn_size,  # actual scalar equation count
    }


def run_subprocess_phases(label, out_dir, mod_name):
    """Cold compile, hot F/J, NR end-to-end — all in fresh subprocess."""
    _clean_numba_cache_for_module(out_dir)

    driver = f'''
import os, sys, time, gc
sys.path.insert(0, {out_dir!r})
gc.disable()

# --- Phase 4: cold import + first-call JIT compile ---
t0 = time.perf_counter()
import {mod_name} as M
t_import = time.perf_counter() - t0

mdl = M.mdl
y = M.y
p = mdl.p

# Force first call (rendered __init__.py already calls F/J during compile,
# but we time it again to be explicit).
t0 = time.perf_counter()
mdl.F(y, p)
mdl.J(y, p)
t_first_call = time.perf_counter() - t0

# Total cold compile time = import + first-call JIT
t_cold = t_import + t_first_call

# --- Phase 6: hot F / J ---
def hot(call, *args, n_warm=10, n_meas=500):
    for _ in range(n_warm):
        call(*args)
    t0 = time.perf_counter()
    for _ in range(n_meas):
        call(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / n_meas

t_F = hot(mdl.F, y, p)
t_J = hot(mdl.J, y, p)

# --- Phase 7: NR end-to-end ---
# Perturb the initial Vm/Va so NR actually iterates instead of
# converging in 1 step from the (already-converged) y stored in
# the rendered module.
import copy as _copy
import numpy as np
from Solverz import nr_method

def _perturbed_y(yfresh):
    y2 = _copy.deepcopy(yfresh)
    # Access per-var slices via the Vars __getitem__ by name.
    try:
        y2['Vm'] = y2['Vm'] + 0.05
    except (KeyError, AttributeError):
        pass
    try:
        y2['Va'] = y2['Va'] + 0.02
    except (KeyError, AttributeError):
        pass
    return y2

# Warm NR once to make sure all helpers compile.
sol_warm = nr_method(mdl, _perturbed_y(M.y))

# Time a fresh NR.
y_run = _perturbed_y(M.y)
t0 = time.perf_counter()
sol = nr_method(M.mdl, y_run)
t_nr = time.perf_counter() - t0

print("COLD_TOTAL", repr(t_cold))
print("HOT_F",     repr(t_F))
print("HOT_J",     repr(t_J))
print("NR_E2E",    repr(t_nr))
print("NR_OK",     int(bool(sol.stats.succeed)))
print("NR_ITS",    int(sol.stats.nstep))
'''
    proc = subprocess.run(
        [sys.executable, '-c', driver],
        capture_output=True, text=True,
        env={**os.environ,
             'NUMBA_CACHE_DIR': '/tmp/empty_numba_cache_' + label},
    )
    if proc.returncode != 0:
        print(f"[{label}] subprocess failed:")
        print("--- stdout ---")
        print(proc.stdout)
        print("--- stderr ---")
        print(proc.stderr)
        return {
            'cold': None, 'hot_F': None, 'hot_J': None,
            'nr': None, 'nr_ok': None, 'nr_its': None,
        }
    out = proc.stdout
    res = {}
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, val = parts
        if key == 'COLD_TOTAL':   res['cold']   = float(eval(val))
        elif key == 'HOT_F':      res['hot_F']  = float(eval(val))
        elif key == 'HOT_J':      res['hot_J']  = float(eval(val))
        elif key == 'NR_E2E':     res['nr']     = float(eval(val))
        elif key == 'NR_OK':      res['nr_ok']  = bool(int(val))
        elif key == 'NR_ITS':     res['nr_its'] = int(val)
    return res


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    print("Benchmarking eps_network loopeqn=False vs loopeqn=True on case30")
    print(f"Output root: {OUT_ROOT}\n")

    print("--- loopeqn=False (legacy per-bus scalar Eqns) ---")
    r_off = run_in_process_phases('loopeqn_off', loopeqn=False)
    print(f"  build={r_off['build']:.3f}s  "
          f"create_instance={r_off['create_instance']:.3f}s  "
          f"render={r_off['render']:.3f}s  "
          f"#njit={r_off['njit_count']}  "
          f"#eqn-families={r_off['n_eqn_families']} "
          f"(eqn_size={r_off['eqn_size']})")
    r_off_sub = run_subprocess_phases('loopeqn_off',
                                      r_off['module_dir'],
                                      r_off['mod_name'])
    r_off.update(r_off_sub)

    print("\n--- loopeqn=True (LoopEqn template) ---")
    r_on = run_in_process_phases('loopeqn_on', loopeqn=True)
    print(f"  build={r_on['build']:.3f}s  "
          f"create_instance={r_on['create_instance']:.3f}s  "
          f"render={r_on['render']:.3f}s  "
          f"#njit={r_on['njit_count']}  "
          f"#eqn-families={r_on['n_eqn_families']} "
          f"(eqn_size={r_on['eqn_size']})")
    r_on_sub = run_subprocess_phases('loopeqn_on',
                                     r_on['module_dir'],
                                     r_on['mod_name'])
    r_on.update(r_on_sub)

    # --- Summary table ---
    print()
    print("=" * 73)
    head = f"{'':<28}{'loopeqn=False':>15}{'loopeqn=True':>15}{'ratio':>12}"
    print(head)
    print("-" * 73)

    def _fmt_s(x):
        if x is None:
            return f"{'FAIL':>14}"
        # Use 4 sig figs for small values so NR end-to-end shows
        # something other than 0.000.
        if x < 0.01:
            return f"{x*1000:>9.3f} ms"
        return f"{x:>12.3f} s"

    def _fmt_us(x):
        return f"{x*1e6:>10.2f} us" if x is not None else f"{'FAIL':>14}"

    def _fmt_int(x):
        return f"{x:>14d}" if x is not None else f"{'FAIL':>14}"

    def _ratio(a, b):
        if a is None or b is None or a == 0:
            return 'n/a'
        return f"{b/a:>10.2f}x"

    rows = [
        ('Model build',           r_off['build'],            r_on['build'],            'sec'),
        ('create_instance',       r_off['create_instance'],  r_on['create_instance'],  'sec'),
        ('render jit=True',       r_off['render'],           r_on['render'],           'sec'),
        ('Numba JIT first',       r_off.get('cold'),         r_on.get('cold'),         'sec'),
        ('@njit function count',  r_off['njit_count'],       r_on['njit_count'],       'int'),
        ('F eval (500-avg)',      r_off.get('hot_F'),        r_on.get('hot_F'),        'us'),
        ('J eval (500-avg)',      r_off.get('hot_J'),        r_on.get('hot_J'),        'us'),
        ('NR end-to-end',         r_off.get('nr'),           r_on.get('nr'),           'sec'),
    ]
    for label, a, b, unit in rows:
        if unit == 'sec':
            a_s = _fmt_s(a); b_s = _fmt_s(b)
        elif unit == 'us':
            a_s = _fmt_us(a); b_s = _fmt_us(b)
        else:
            a_s = _fmt_int(a); b_s = _fmt_int(b)
        print(f"{label:<28}{a_s:>15}{b_s:>15}{_ratio(a, b):>12}")
    print("=" * 73)
    print()
    print(f"#scalar equations (eqn_size): loopeqn=False={r_off['eqn_size']}, "
          f"loopeqn=True={r_on['eqn_size']}")
    print(f"#equation families: loopeqn=False={r_off['n_eqn_families']}, "
          f"loopeqn=True={r_on['n_eqn_families']}")
    if r_off.get('nr_ok') is not None:
        print(f"NR converged — off: {r_off['nr_ok']} "
              f"({r_off.get('nr_its')} steps); "
              f"on: {r_on['nr_ok']} ({r_on.get('nr_its')} steps)")
    print(f"Artifacts at: {OUT_ROOT}")


if __name__ == '__main__':
    main()
