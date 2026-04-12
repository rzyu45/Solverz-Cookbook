"""Benchmark Mat_Mul vs non-Mat_Mul power-flow formulations on case30.

Phases measured (each reported separately so costs don't bleed into
each other):

  1. Model construction         — ``Model()`` + ``create_instance()``
  2. FormJac                    — ``spf.FormJac(y0)`` (sparse pattern
                                   analysis, SpDiag perturbation for
                                   the Mat_Mul mutable-matrix blocks)
  3. Inline compile             — ``made_numerical(spf, y0, sparse)``
                                   (sympy lambdify — no Numba)
  4. Inline hot F / J           — steady-state per-call time
  5. Module render              — ``module_printer(...).render()``
                                   (emits the .py file + pickle)
  6. Module cold import+compile — fresh subprocess with __pycache__
                                   and numba .nbi/.nbc cleared, so
                                   the import triggers full @njit
                                   compilation
  7. Module hot F / J           — steady-state per-call time inside
                                   the same subprocess

Prints a summary table at the end.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Output directory for the rendered modules (persistent across phases,
# cleaned up at the end).
OUT_ROOT = tempfile.mkdtemp(prefix="bench_pf_")

# Resolve case30 data directory relative to this script so the
# benchmark is location-independent. The .mat files live next to
# pf_mdl.py / pf_matmul.py under ``test_pf_jac/``.
CASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_pf_jac')

# ---------------------------------------------------------------------------
# Model builders (separate functions so each can be timed in isolation).
# ---------------------------------------------------------------------------

def build_polar():
    import numpy as np
    from scipy.io import loadmat
    from Solverz import Var, Eqn, Model, sin, cos, Param

    sys_data = loadmat(os.path.join(CASE_DATA_DIR, 'pf.mat'))
    PQ = loadmat(os.path.join(CASE_DATA_DIR, 'pq.mat'))

    V = sys_data["V"].reshape((-1,))
    nb = V.shape[0]
    Ybus = sys_data["Ybus"]
    G = Ybus.real.toarray()
    B = Ybus.imag.toarray()
    ref = (sys_data["ref"] - 1).reshape((-1,)).tolist()
    pv = (sys_data["pv"] - 1).reshape((-1,)).tolist()
    pq = (sys_data["pq"] - 1).reshape((-1,)).tolist()
    mbase = 100.0

    m = Model()
    m.Va = Var("Va", np.angle(V)[pv + pq])
    m.Vm = Var("Vm", np.abs(V)[pq])
    m.Pg = Param("Pg", PQ["Pg"].reshape(-1) / mbase)
    m.Qg = Param("Qg", PQ["Qg"].reshape(-1) / mbase)
    m.Pd = Param("Pd", PQ["Pd"].reshape(-1) / mbase)
    m.Qd = Param("Qd", PQ["Qd"].reshape(-1) / mbase)

    def get_Vm(idx):
        if idx in ref + pv:
            return np.abs(V)[idx]
        elif idx in pq:
            return m.Vm[pq.index(idx)]

    def get_Va(idx):
        if idx in ref:
            return np.angle(V)[idx]
        elif idx in pv + pq:
            return m.Va[(pv + pq).index(idx)]

    for i in pv + pq:
        expr = 0
        Vmi, Vai = get_Vm(i), get_Va(i)
        for j in range(nb):
            Vmj, Vaj = get_Vm(j), get_Va(j)
            expr += Vmi * Vmj * (G[i, j] * cos(Vai - Vaj) + B[i, j] * sin(Vai - Vaj))
        m.__dict__[f"P_eqn_{i}"] = Eqn(f"P_eqn_{i}", expr + m.Pd[i] - m.Pg[i])

    for i in pq:
        expr = 0
        Vmi, Vai = get_Vm(i), get_Va(i)
        for j in range(nb):
            Vmj, Vaj = get_Vm(j), get_Va(j)
            expr += Vmi * Vmj * (G[i, j] * sin(Vai - Vaj) - B[i, j] * cos(Vai - Vaj))
        m.__dict__[f"Q_eqn_{i}"] = Eqn(f"Q_eqn_{i}", expr + m.Qd[i] - m.Qg[i])

    spf, y0 = m.create_instance()
    return spf, y0


def build_matmul():
    import numpy as np
    from scipy.io import loadmat
    from scipy.sparse import csc_array
    from Solverz import Var, Eqn, Model, Param, Mat_Mul

    sys_data = loadmat(os.path.join(CASE_DATA_DIR, 'pf.mat'))
    PQ = loadmat(os.path.join(CASE_DATA_DIR, 'pq.mat'))

    V = sys_data["V"].reshape((-1,))
    nb = V.shape[0]
    Ybus = sys_data["Ybus"].tocsc()
    G_full = Ybus.real
    B_full = Ybus.imag
    ref = (sys_data["ref"] - 1).reshape((-1,)).tolist()
    pv = (sys_data["pv"] - 1).reshape((-1,)).tolist()
    pq = (sys_data["pq"] - 1).reshape((-1,)).tolist()
    non_ref = pv + pq
    mbase = 100.0

    e0 = V.real
    f0 = V.imag
    Pg = PQ["Pg"].reshape(-1) / mbase
    Qg = PQ["Qg"].reshape(-1) / mbase
    Pd = PQ["Pd"].reshape(-1) / mbase
    Qd = PQ["Qd"].reshape(-1) / mbase
    Pinj = Pg - Pd
    Qinj = Qg - Qd

    n_nr = len(non_ref)
    G_nr = csc_array(G_full[np.ix_(non_ref, non_ref)])
    B_nr = csc_array(B_full[np.ix_(non_ref, non_ref)])

    e_ref = e0[ref[0]]
    f_ref = f0[ref[0]]
    G_ref_col = G_full[non_ref, ref[0]].toarray().ravel()
    B_ref_col = B_full[non_ref, ref[0]].toarray().ravel()
    p_ref = G_ref_col * e_ref - B_ref_col * f_ref
    q_ref = B_ref_col * e_ref + G_ref_col * f_ref

    pq_in_nr = [non_ref.index(i) for i in pq]
    G_pq = csc_array(G_full[np.ix_(pq, non_ref)])
    B_pq = csc_array(B_full[np.ix_(pq, non_ref)])
    G_pq_ref_col = G_full[pq, ref[0]].toarray().ravel()
    B_pq_ref_col = B_full[pq, ref[0]].toarray().ravel()
    p_ref_pq = G_pq_ref_col * e_ref - B_pq_ref_col * f_ref
    q_ref_pq = B_pq_ref_col * e_ref + G_pq_ref_col * f_ref

    m = Model()
    m.e = Var("e", np.ones(n_nr))
    m.f = Var("f", np.zeros(n_nr))
    m.G_nr = Param("G_nr", G_nr, dim=2, sparse=True)
    m.B_nr = Param("B_nr", B_nr, dim=2, sparse=True)
    m.G_pq = Param("G_pq", G_pq, dim=2, sparse=True)
    m.B_pq = Param("B_pq", B_pq, dim=2, sparse=True)
    m.p_ref = Param("p_ref", p_ref)
    m.q_ref = Param("q_ref", q_ref)
    m.p_ref_pq = Param("p_ref_pq", p_ref_pq)
    m.q_ref_pq = Param("q_ref_pq", q_ref_pq)
    m.Pinj = Param("Pinj", Pinj[non_ref])
    m.Qinj = Param("Qinj", Qinj[pq])

    m.P_eqn = Eqn("P_balance",
                  m.e * (Mat_Mul(m.G_nr, m.e) - Mat_Mul(m.B_nr, m.f) + m.p_ref)
                  + m.f * (Mat_Mul(m.B_nr, m.e) + Mat_Mul(m.G_nr, m.f) + m.q_ref)
                  - m.Pinj)

    e_pq = m.e[pq_in_nr[0]:pq_in_nr[-1] + 1]
    f_pq = m.f[pq_in_nr[0]:pq_in_nr[-1] + 1]
    m.Q_eqn = Eqn("Q_balance",
                  f_pq * (Mat_Mul(m.G_pq, m.e) - Mat_Mul(m.B_pq, m.f) + m.p_ref_pq)
                  - e_pq * (Mat_Mul(m.B_pq, m.e) + Mat_Mul(m.G_pq, m.f) + m.q_ref_pq)
                  - m.Qinj)

    pv_in_nr = [non_ref.index(i) for i in pv]
    Vm_pv_sq = (abs(V[pv])) ** 2
    m.Vm_sq = Param("Vm_sq", Vm_pv_sq)
    e_pv = m.e[pv_in_nr[0]:pv_in_nr[-1] + 1]
    f_pv = m.f[pv_in_nr[0]:pv_in_nr[-1] + 1]
    m.V_eqn = Eqn("V_pv", e_pv ** 2 + f_pv ** 2 - m.Vm_sq)

    spf, y0 = m.create_instance()
    return spf, y0


# ---------------------------------------------------------------------------
# Timing helpers.
# ---------------------------------------------------------------------------

def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, t1 - t0


def _time_hot(call, *args, n_warm=5, n_meas=200):
    for _ in range(n_warm):
        call(*args)
    t0 = time.perf_counter()
    for _ in range(n_meas):
        call(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / n_meas


def _time_model_build(build_fn):
    """Build phase without FormJac — just Model + create_instance."""
    # We need to split create_instance from FormJac. Model() + eqn
    # construction happens in build_fn up to create_instance() call.
    # Since create_instance *itself* is the expensive step we time it
    # end-to-end (cheap cost attribution isn't worth the refactor).
    return _timed(build_fn)


def _clean_numba_cache_for_module(module_dir):
    """Delete __pycache__ and .nbi/.nbc files so the next import
    triggers a cold numba compile."""
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


# ---------------------------------------------------------------------------
# Phase runners (in-process for phases 1–5, subprocess for phase 6).
# ---------------------------------------------------------------------------

def run_modeling_phases(kind, build_fn):
    """Phases 1–5 in-process, plus Phase 7 hot F/J via module import."""
    results = {}

    (spf, y0), t_build = _time_model_build(build_fn)
    results['1_build'] = t_build

    _, t_formjac = _timed(spf.FormJac, y0)
    results['2_formjac'] = t_formjac

    from Solverz import made_numerical
    mdl, t_inline = _timed(made_numerical, spf, y0, sparse=True)
    results['3_inline_compile'] = t_inline

    # Hot F/J for inline. First evaluation may pay a cache-compile cost
    # under lambdify, so do a few warm calls before measuring.
    t_inline_F = _time_hot(mdl.F, y0, mdl.p)
    t_inline_J = _time_hot(mdl.J, y0, mdl.p)
    results['4_inline_F'] = t_inline_F
    results['4_inline_J'] = t_inline_J

    # Module render
    from Solverz import module_printer
    out_dir = os.path.join(OUT_ROOT, kind)
    os.makedirs(out_dir, exist_ok=True)
    _, t_render = _timed(lambda: module_printer(
        spf, y0, f'pf_{kind}_mod', directory=out_dir, jit=True).render())
    results['5_render'] = t_render
    results['module_dir'] = out_dir
    return results


def run_module_cold_and_hot(kind, out_dir):
    """Phase 6: cold compile (fresh subprocess, numba cache cleared),
    plus Phase 7: hot F/J in that same subprocess.

    We use a subprocess so that:
    - The numba compile isn't skewed by already-imported numba state.
    - We can cleanly wipe __pycache__ + .nbi/.nbc first.
    """
    _clean_numba_cache_for_module(out_dir)

    driver = f'''
import os
import sys
import time
sys.path.insert(0, {out_dir!r})

import gc
gc.disable()

t0 = time.perf_counter()
import pf_{kind}_mod as M
t_import = time.perf_counter() - t0

mdl = M.mdl
y = M.y
p = mdl.p

# The module's __init__.py already calls mdl.F / mdl.J once as part of
# its "compile" step (to warm numba). So by the time we get here, the
# @njit caches are hot. Report the total import time (which includes
# that warm-up) as the cold-compile cost.

# Hot F/J — average over 200 calls.
def hot(call, *args, n_warm=5, n_meas=200):
    for _ in range(n_warm):
        call(*args)
    t0 = time.perf_counter()
    for _ in range(n_meas):
        call(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / n_meas

t_F = hot(mdl.F, y, p)
t_J = hot(mdl.J, y, p)

print(f"COLD_IMPORT {{t_import:.6f}}")
print(f"HOT_F {{t_F:.9f}}")
print(f"HOT_J {{t_J:.9f}}")
'''
    proc = subprocess.run(
        [sys.executable, '-c', driver],
        capture_output=True, text=True,
        env={**os.environ, 'NUMBA_CACHE_DIR': '/tmp/empty_numba_cache_' + kind},
    )
    if proc.returncode != 0:
        print(f"[{kind}] subprocess failed: {proc.stderr}")
        return {'6_cold_import': None, '7_module_F': None, '7_module_J': None}
    out = proc.stdout
    results = {}
    for line in out.splitlines():
        if line.startswith('COLD_IMPORT '):
            results['6_cold_import'] = float(line.split()[1])
        elif line.startswith('HOT_F '):
            results['7_module_F'] = float(line.split()[1])
        elif line.startswith('HOT_J '):
            results['7_module_J'] = float(line.split()[1])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building and profiling BOTH formulations on case30...")
    print(f"Output root: {OUT_ROOT}")
    print()

    print("--- polar (no Mat_Mul) ---")
    r_polar = run_modeling_phases('polar', build_polar)
    r_polar_cold = run_module_cold_and_hot('polar', r_polar['module_dir'])
    r_polar.update(r_polar_cold)

    print("--- rectangular + Mat_Mul ---")
    r_mm = run_modeling_phases('matmul', build_matmul)
    r_mm_cold = run_module_cold_and_hot('matmul', r_mm['module_dir'])
    r_mm.update(r_mm_cold)

    # Print comparison table
    print()
    print("=" * 72)
    print(f"{'Phase':<36} {'polar':>14} {'Mat_Mul':>14} {'ratio':>6}")
    print("-" * 72)
    labels = [
        ('1. Model + create_instance',  '1_build',          's'),
        ('2. FormJac',                  '2_formjac',        's'),
        ('3. Inline compile',           '3_inline_compile', 's'),
        ('4. Inline hot F (per call)',  '4_inline_F',       'us'),
        ('4. Inline hot J (per call)',  '4_inline_J',       'us'),
        ('5. Module render',            '5_render',         's'),
        ('6. Module cold import+JIT',   '6_cold_import',    's'),
        ('7. Module hot F (per call)',  '7_module_F',       'us'),
        ('7. Module hot J (per call)',  '7_module_J',       'us'),
    ]
    for label, key, unit in labels:
        a = r_polar.get(key)
        b = r_mm.get(key)
        if a is None or b is None:
            print(f"{label:<36} {'FAIL':>14} {'FAIL':>14}")
            continue
        if unit == 'us':
            a_str = f"{a * 1e6:>10.2f} us"
            b_str = f"{b * 1e6:>10.2f} us"
        else:
            a_str = f"{a:>11.3f} s"
            b_str = f"{b:>11.3f} s"
        ratio = b / a if a > 0 else float('nan')
        print(f"{label:<36} {a_str:>14} {b_str:>14} {ratio:>5.2f}x")
    print("=" * 72)
    print()
    print(f"Shape — polar: {53} unknowns / 53 eqns")
    print(f"Shape — Mat_Mul: 58 unknowns / 58 eqns (e+f at non-ref + V² at PV)")
    print()
    print(f"Artifacts preserved at: {OUT_ROOT}")


if __name__ == '__main__':
    main()
