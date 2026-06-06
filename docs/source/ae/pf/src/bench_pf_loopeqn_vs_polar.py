"""Benchmark ``LoopEqn`` vs the traditional for-loop power-flow
formulation on case30.

Both formulations stay in **polar** coordinates and solve the same
case30 power flow. The only difference is how the per-bus power-balance
summations are handed to Solverz:

  * **for-loop** (``pf_mdl.py``) — Python ``for`` loops expand each
    bus's power balance into one scalar ``Eqn``. case30 produces 53
    scalar ``Eqn``s (P at pv+pq, Q at pq).
  * **LoopEqn** (``pf_mdl_loopeqn.py``) — one symbolic ``LoopEqn`` per
    balance family iterates the outer bus index over a ``Set`` and sums
    over the neighbour ``Set``. case30 produces 4 equation families
    (P_eqn, Q_eqn, Vm_pin, Va_pin), 60 scalar rows after pinning the
    ref/pv buses on the flat ``Vm_full`` / ``Va_full`` state.

``made_numerical`` (the inline lambdify path) does **not** support
LoopEqn — it is designed for the Numba-JIT module-printer path (see
Solverz issue #132). The comparison is therefore module-path-only:
every phase below runs on the ``module_printer(..., jit=True)``
pipeline, which is the production path anyway.

Phases measured (mapped to the three axes the cookbook reports):

  modelling   1. Model build + ``create_instance``
  compilation 2. Module render (``module_printer(..., jit=True).render``)
              3. ``@njit`` kernel count (every ``@njit`` in num_func.py,
                 not just the ``inner_*`` ones)
              4. Module cold import + JIT compile (fresh subprocess,
                 ``__pycache__`` and Numba ``.nbi``/``.nbc`` wiped; the
                 rendered ``__init__`` warms F/J during import, so the
                 import time IS the cold-compile cost)
  computation 5. Module hot F / J (steady-state per-call time)
              6. Newton-Raphson end-to-end (``nr_method``)

Prints a summary table at the end. Re-run on any hardware with::

    cd docs/source/ae/pf/src
    python bench_pf_loopeqn_vs_polar.py
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Persistent output dir for the rendered modules (left on disk so the
# numbers are inspectable; the path is printed at the end).
OUT_ROOT = tempfile.mkdtemp(prefix="bench_pf_loopeqn_")

CASE_DATA_DIR = os.path.join(HERE, 'test_pf_jac')


# ---------------------------------------------------------------------------
# Model builders. Each returns ``(spf, y0)`` so phase 1 times the whole
# "Model() + equations + create_instance()" modelling cost identically.
# ---------------------------------------------------------------------------

def build_polar():
    """Traditional for-loop polar model (identical to ``pf_mdl.py``)."""
    # Re-use the canonical polar builder that already backs the
    # Mat_Mul-vs-polar comparison so the two benchmarks measure the
    # exact same for-loop model.
    from bench_pf_matmul_vs_polar import build_polar as _bp
    return _bp()


def build_loopeqn():
    """LoopEqn polar model (from ``pf_mdl_loopeqn.py``)."""
    from pf_mdl_loopeqn import build_loopeqn_pf_model
    m = build_loopeqn_pf_model(datadir=CASE_DATA_DIR)
    return m.create_instance()


# ---------------------------------------------------------------------------
# Timing helpers.
# ---------------------------------------------------------------------------

def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, t1 - t0


def _clean_numba_cache_for_module(module_dir):
    """Delete ``__pycache__`` and ``.nbi``/``.nbc`` so the next import
    triggers a cold Numba compile."""
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


def _count_njit_kernels(num_func_path):
    """Count every ``@njit``-decorated function in num_func.py.

    This is the honest compile-surface metric: it counts ALL kernels
    Numba must compile, not just the ``inner_*`` ones. The for-loop
    module emits only ``inner_F*`` / ``inner_J*`` kernels, but the
    LoopEqn module also emits ``_sz_loop_jac_kernel_N`` (per-family
    Jacobian assembly) and ``_sz_csr_*_point`` (CSR element lookup)
    helpers, each of which is ``@njit(cache=True)`` and pays the same
    per-kernel compile cost. Counting only ``inner_*`` would compare a
    full count on one side against a filtered subset on the other.
    """
    if not os.path.exists(num_func_path):
        return None
    with open(num_func_path, 'r') as f:
        text = f.read()
    return len(re.findall(r'^@njit', text, flags=re.MULTILINE))


# ---------------------------------------------------------------------------
# Phase runners.
# ---------------------------------------------------------------------------

def run_modeling_phases(kind, build_fn):
    """Phases 1-3 in-process: build + create_instance, render, njit count."""
    from Solverz import module_printer

    (spf, y0), t_build = _timed(build_fn)

    out_dir = os.path.join(OUT_ROOT, kind)
    os.makedirs(out_dir, exist_ok=True)
    mod_name = f'pf_{kind}_mod'
    _, t_render = _timed(lambda: module_printer(
        spf, y0, mod_name, directory=out_dir, jit=True).render())

    num_func_path = os.path.join(out_dir, mod_name, 'num_func.py')
    n_njit = _count_njit_kernels(num_func_path)

    return {
        'build': t_build,
        'render': t_render,
        'njit_count': n_njit,
        'module_dir': out_dir,
        'mod_name': mod_name,
        'n_eqn_families': len(spf.EQNs),
        'eqn_size': spf.eqn_size,
    }


def run_subprocess_phases(kind, out_dir, mod_name):
    """Phases 4-6 in a fresh subprocess with the Numba cache wiped:
    cold import+JIT, hot F/J, NR end-to-end."""
    _clean_numba_cache_for_module(out_dir)

    driver = f'''
import os, sys, time, gc, copy
sys.path.insert(0, {out_dir!r})
gc.disable()

# --- Phase 4: cold import + JIT compile ---
# The rendered __init__ already calls F/J once during import to warm
# the Numba caches, so the import time IS the cold-compile cost (this
# matches how bench_pf_matmul_vs_polar.py reports COLD).
t0 = time.perf_counter()
import {mod_name} as M
t_cold = time.perf_counter() - t0

mdl = M.mdl
y = M.y
p = mdl.p

# --- Phase 5: hot F / J (steady state) ---
def hot(call, *args, n_warm=10, n_meas=2000):
    for _ in range(n_warm):
        call(*args)
    t0 = time.perf_counter()
    for _ in range(n_meas):
        call(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / n_meas

t_F = hot(mdl.F, y, p)
t_J = hot(mdl.J, y, p)

# --- Phase 6: Newton-Raphson end-to-end ---
# Perturb the (already-converged) stored y so NR actually iterates.
# NOTE: the two models parameterise their state differently — the
# for-loop model has 53 free vars (Vm at pq, Va at pv+pq) while the
# LoopEqn model has 60 (full Vm/Va with ref/pv pinned). Each model is
# perturbed only on its OWN vars below, so the perturbation scope (and
# hence the NR step count) differs. Read the NR-end-to-end row together
# with the printed step counts, not as a pure per-iteration comparison.
from Solverz import nr_method

def _perturbed(yfresh):
    y2 = type(yfresh)(yfresh.a, yfresh.array.copy())
    for name, dv in (('Vm', 0.05), ('Va', 0.02),
                     ('Vm_full', 0.05), ('Va_full', 0.02)):
        if name in y2.var_list:
            y2[name] = y2[name] + dv
    return y2

nr_method(mdl, _perturbed(M.y))           # warm
t0 = time.perf_counter()
sol = nr_method(mdl, _perturbed(M.y))
t_nr = time.perf_counter() - t0

print("COLD", repr(t_cold))
print("HOT_F", repr(t_F))
print("HOT_J", repr(t_J))
print("NR", repr(t_nr))
print("NR_OK", int(bool(sol.stats.succeed)))
print("NR_ITS", int(sol.stats.nstep))
'''
    # Point Numba at a fresh cache dir under OUT_ROOT (itself a per-run
    # mkdtemp), so the cold-compile measurement can never be served a
    # stale cache from a previous run.
    nb_cache = os.path.join(OUT_ROOT, 'nbcache_' + kind)
    os.makedirs(nb_cache, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, '-c', driver],
        capture_output=True, text=True,
        env={**os.environ, 'NUMBA_CACHE_DIR': nb_cache},
    )
    if proc.returncode != 0:
        print(f"[{kind}] subprocess failed:")
        print(proc.stdout)
        print(proc.stderr)
        return {'cold': None, 'hot_F': None, 'hot_J': None,
                'nr': None, 'nr_ok': None, 'nr_its': None}
    res = {}
    for line in proc.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, val = parts
        if key == 'COLD':     res['cold'] = float(eval(val))
        elif key == 'HOT_F':  res['hot_F'] = float(eval(val))
        elif key == 'HOT_J':  res['hot_J'] = float(eval(val))
        elif key == 'NR':     res['nr'] = float(eval(val))
        elif key == 'NR_OK':  res['nr_ok'] = bool(int(val))
        elif key == 'NR_ITS': res['nr_its'] = int(val)
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Benchmarking LoopEqn vs for-loop polar power flow on case30")
    print(f"Output root: {OUT_ROOT}\n")

    print("--- for-loop (per-bus scalar Eqns; pf_mdl.py) ---")
    r_loop_off = run_modeling_phases('forloop', build_polar)
    print(f"  build={r_loop_off['build']:.3f}s  "
          f"render={r_loop_off['render']:.3f}s  "
          f"#njit={r_loop_off['njit_count']}  "
          f"#eqn-families={r_loop_off['n_eqn_families']} "
          f"(eqn_size={r_loop_off['eqn_size']})")
    r_loop_off.update(run_subprocess_phases(
        'forloop', r_loop_off['module_dir'], r_loop_off['mod_name']))

    print("\n--- LoopEqn (symbolic Set/Sum template; pf_mdl_loopeqn.py) ---")
    r_loop_on = run_modeling_phases('loopeqn', build_loopeqn)
    print(f"  build={r_loop_on['build']:.3f}s  "
          f"render={r_loop_on['render']:.3f}s  "
          f"#njit={r_loop_on['njit_count']}  "
          f"#eqn-families={r_loop_on['n_eqn_families']} "
          f"(eqn_size={r_loop_on['eqn_size']})")
    r_loop_on.update(run_subprocess_phases(
        'loopeqn', r_loop_on['module_dir'], r_loop_on['mod_name']))

    # --- Summary table ---
    print()
    print("=" * 74)
    print(f"{'Phase':<32}{'for-loop':>14}{'LoopEqn':>14}{'ratio':>12}")
    print("-" * 74)

    def _fmt_s(x):
        if x is None:
            return f"{'FAIL':>12}"
        if x < 0.01:
            return f"{x*1000:>9.3f} ms"
        return f"{x:>11.3f} s"

    def _fmt_us(x):
        return f"{x*1e6:>10.2f} us" if x is not None else f"{'FAIL':>12}"

    def _fmt_int(x):
        return f"{x:>12d}" if x is not None else f"{'FAIL':>12}"

    def _ratio(a, b):
        # report how much LoopEqn beats (or loses to) for-loop: a=for-loop, b=loopeqn
        if a is None or b is None or b == 0:
            return 'n/a'
        return f"{a/b:>9.2f}x"

    rows = [
        ('1. Model + create_instance', r_loop_off['build'],          r_loop_on['build'],          'sec'),
        ('2. Module render (jit=True)', r_loop_off['render'],         r_loop_on['render'],         'sec'),
        ('3. @njit kernel count',       r_loop_off['njit_count'],     r_loop_on['njit_count'],     'int'),
        ('4. Module cold import+JIT',   r_loop_off.get('cold'),       r_loop_on.get('cold'),       'sec'),
        ('5. Module hot F (per call)',  r_loop_off.get('hot_F'),      r_loop_on.get('hot_F'),      'us'),
        ('5. Module hot J (per call)',  r_loop_off.get('hot_J'),      r_loop_on.get('hot_J'),      'us'),
        ('6. NR end-to-end',            r_loop_off.get('nr'),         r_loop_on.get('nr'),         'sec'),
    ]
    for label, a, b, unit in rows:
        if unit == 'sec':
            a_s, b_s = _fmt_s(a), _fmt_s(b)
        elif unit == 'us':
            a_s, b_s = _fmt_us(a), _fmt_us(b)
        else:
            a_s, b_s = _fmt_int(a), _fmt_int(b)
        print(f"{label:<32}{a_s:>14}{b_s:>14}{_ratio(a, b):>12}")
    print("=" * 74)
    print("ratio = for-loop / LoopEqn  (>1 means LoopEqn is faster/smaller)\n")
    print(f"#scalar equations (eqn_size): for-loop={r_loop_off['eqn_size']}, "
          f"LoopEqn={r_loop_on['eqn_size']}")
    print(f"#equation families: for-loop={r_loop_off['n_eqn_families']}, "
          f"LoopEqn={r_loop_on['n_eqn_families']}")
    if r_loop_off.get('nr_ok') is not None:
        print(f"NR converged — for-loop: {r_loop_off['nr_ok']} "
              f"({r_loop_off.get('nr_its')} steps); "
              f"LoopEqn: {r_loop_on['nr_ok']} ({r_loop_on.get('nr_its')} steps)")
    print(f"Artifacts at: {OUT_ROOT}")


if __name__ == '__main__':
    main()
