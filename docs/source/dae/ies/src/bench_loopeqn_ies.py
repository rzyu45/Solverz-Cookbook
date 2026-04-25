"""Benchmark cookbook IES model: loopeqn=False vs loopeqn=True.

Compares model build, codegen, Numba JIT first-call compile time, the
@njit function-count proxy for compile workload, and steady-state
F/J evaluation latency for the two model paths.

Usage
-----
Run from any directory (paths are absolute)::

    python bench_loopeqn_ies.py

The rendered modules are dropped into a tmp directory under
``out_dir`` next to this script and removed after the run.

The script reuses ``_build_ies_model`` from the sibling
``test_ies.py`` so both paths exercise exactly the same model
topology — only the ``loopeqn`` flag on ``heat_network``,
``gas_network``, and (where supported) ``eps_network`` differs.
"""
import gc
import importlib
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Import the existing model builder from test_ies.py without going
# through pytest. The module lives next to this script.
HERE = Path(__file__).resolve().parent
DATADIR = HERE / "test_ies"
sys.path.insert(0, str(HERE))
import test_ies  # noqa: E402

from Solverz import Rodas, Opt, module_printer  # noqa: E402

ITERS_HOT = 500
# Short Rodas horizon for correctness verification — same warmup as
# test_ies.py but truncated to keep the benchmark tractable. Long enough
# to exercise the algebraic + differential structure.
VERIFY_T_END = 300.0  # seconds


def _count_njit_functions(num_func_path: Path) -> int:
    """Count ``def inner_*`` definitions in the rendered ``num_func.py``.

    This is the proxy for "@njit function workload" requested in the
    task — every ``inner_F<N>`` / ``inner_J<N>`` lambda is wrapped in
    its own ``@njit(cache=True)`` decorator by the module generator.
    """
    if not num_func_path.exists():
        return 0
    text = num_func_path.read_text()
    return len(re.findall(r"^def inner_", text, flags=re.MULTILINE))


def _hot_eval(callable_, args, n_iters):
    """Average wall time per call after a warmup run."""
    callable_(*args)  # warmup (already JIT-compiled, but warm caches)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        callable_(*args)
    return (time.perf_counter() - t0) / n_iters


def bench_one(loopeqn: bool, tag: str, out_root: Path):
    """Run one full benchmark pass for a given ``loopeqn`` setting."""
    print(f"\n{'=' * 70}")
    print(f"  Benchmark pass: loopeqn={loopeqn}  (tag={tag})")
    print(f"{'=' * 70}")

    # ---- 1. Model build ------------------------------------------------
    gc.collect()
    t0 = time.perf_counter()
    model = test_ies._build_ies_model(DATADIR, loopeqn=loopeqn)
    t_build = time.perf_counter() - t0
    print(f"[{tag}] model build:       {t_build:8.3f} s")

    # ---- 2. create_instance --------------------------------------------
    gc.collect()
    t0 = time.perf_counter()
    sdae, y0 = model.create_instance()
    t_instance = time.perf_counter() - t0
    print(f"[{tag}] create_instance:   {t_instance:8.3f} s")

    # ---- 3. render(jit=True) -------------------------------------------
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    module_name = f"ies_bench_{tag}"

    gc.collect()
    printer = module_printer(sdae, y0, module_name,
                             directory=str(out_dir), jit=True)
    t0 = time.perf_counter()
    printer.render()
    t_render = time.perf_counter() - t0
    print(f"[{tag}] render jit=True:   {t_render:8.3f} s")

    # ---- 4. @njit function count ---------------------------------------
    num_func = out_dir / module_name / "num_func.py"
    n_njit = _count_njit_functions(num_func)
    print(f"[{tag}] @njit functions:   {n_njit:8d}")

    # ---- 5. Numba JIT first-call compile (import) ----------------------
    # Importing the package triggers ``mdl.F(0, y, p)`` and
    # ``mdl.J(0, y, p)`` once at module load — that single round-trip is
    # the user-visible "first-call" Numba compile cost.
    sys.path.insert(0, str(out_dir))
    gc.collect()
    t0 = time.perf_counter()
    mod = importlib.import_module(module_name)
    t_jit_first = time.perf_counter() - t0
    print(f"[{tag}] Numba JIT first:   {t_jit_first:8.3f} s")

    # ---- 6. Hot F / J evaluations --------------------------------------
    mdl = mod.mdl
    p = mod.p
    y = mod.y
    args = (0.0, y, p)

    t_F = _hot_eval(mdl.F, args, ITERS_HOT)
    t_J = _hot_eval(mdl.J, args, ITERS_HOT)
    print(f"[{tag}] F eval ({ITERS_HOT} avg): {t_F * 1e6:9.2f} us")
    print(f"[{tag}] J eval ({ITERS_HOT} avg): {t_J * 1e6:9.2f} us")

    # ---- 7. Correctness verification via short Rodas run ---------------
    print(f"[{tag}] Rodas [0, {VERIFY_T_END}] for correctness check...")
    t0 = time.perf_counter()
    sol = Rodas(mdl, np.linspace(0, VERIFY_T_END, 301), y, Opt(pbar=False))
    t_rodas = time.perf_counter() - t0
    print(f"[{tag}] Rodas: {len(sol.T)} steps, {t_rodas:.2f}s, "
          f"last t={sol.T[-1]:.2f}")

    # Capture final-state dict keyed by variable name.
    final_state = {k: np.asarray(sol.Y[-1][k]).ravel().copy()
                   for k in sol.Y[-1].var_list}

    # ---- cleanup module imports so a second pass does not collide ------
    sys.path.remove(str(out_dir))
    for k in list(sys.modules):
        if k.startswith(module_name):
            del sys.modules[k]
    del mod, mdl

    return {
        "build": t_build,
        "instance": t_instance,
        "render": t_render,
        "jit_first": t_jit_first,
        "n_njit": n_njit,
        "F_us": t_F * 1e6,
        "J_us": t_J * 1e6,
        "rodas_s": t_rodas,
        "rodas_steps": len(sol.T),
        "final_state": final_state,
    }


def _ratio(a, b):
    if b == 0:
        return float("nan")
    return a / b


def print_table(legacy, loop):
    """Final markdown-style comparison table."""
    rows = [
        ("Model build (s)",       legacy["build"],     loop["build"]),
        ("create_instance (s)",   legacy["instance"],  loop["instance"]),
        ("render jit=True (s)",   legacy["render"],    loop["render"]),
        ("Numba JIT first (s)",   legacy["jit_first"], loop["jit_first"]),
        ("@njit function count",  legacy["n_njit"],    loop["n_njit"]),
        ("F eval (us, 500-avg)",  legacy["F_us"],      loop["F_us"]),
        ("J eval (us, 500-avg)",  legacy["J_us"],      loop["J_us"]),
        (f"Rodas [0,{VERIFY_T_END:.0f}] (s)",
                                  legacy["rodas_s"],   loop["rodas_s"]),
    ]
    print()
    print("=" * 73)
    print(f"{'':30s}{'loopeqn=False':>14s}  {'loopeqn=True':>13s}  {'ratio (T/F)':>12s}")
    print("-" * 73)
    for name, lf, lt in rows:
        if "count" in name:
            r = _ratio(lt, lf)
            print(f"{name:30s}{lf:>14.0f}  {lt:>13.0f}  {r:>11.3f}x")
        else:
            r = _ratio(lt, lf)
            print(f"{name:30s}{lf:>14.3f}  {lt:>13.3f}  {r:>11.3f}x")
    print("=" * 73)


def print_accuracy(legacy, loop):
    """Compare final Rodas states between legacy and loop paths."""
    f_state = legacy["final_state"]
    l_state = loop["final_state"]
    shared = sorted(set(f_state) & set(l_state))
    print()
    print(f"Accuracy check: final state at t={VERIFY_T_END:.0f}s")
    print(f"  legacy steps: {legacy['rodas_steps']}, "
          f"loop steps: {loop['rodas_steps']}")
    print(f"  {len(shared)} shared variables")
    worst_abs = 0.0
    worst_rel = 0.0
    worst_var = ''
    for v in shared:
        a = f_state[v]; b = l_state[v]
        if a.shape != b.shape:
            print(f"  SHAPE MISMATCH in {v}: {a.shape} vs {b.shape}")
            continue
        adiff = np.max(np.abs(a - b))
        scale = np.max(np.abs(a)) + 1e-15
        rdiff = adiff / scale
        if rdiff > worst_rel:
            worst_rel = rdiff
            worst_abs = adiff
            worst_var = v
    print(f"  max relative error: {worst_rel:.3e}  "
          f"(var={worst_var!r}, abs={worst_abs:.3e})")
    if worst_rel > 1e-2:
        print(f"  WARNING: relative error {worst_rel:.2e} > 1% threshold!")
    else:
        print(f"  OK: relative error within 1% across all {len(shared)} vars")


def main():
    out_root = Path(tempfile.mkdtemp(prefix="bench_loopeqn_ies_"))
    print(f"Output dir: {out_root}")
    try:
        # Run loopeqn=False FIRST — it is the slow case; if it falls
        # over we still get a partial readout.
        legacy = bench_one(loopeqn=False, tag="legacy", out_root=out_root)
        loop = bench_one(loopeqn=True, tag="loop", out_root=out_root)
        print_table(legacy, loop)
        print_accuracy(legacy, loop)
    finally:
        # Best-effort cleanup of generated modules.
        try:
            shutil.rmtree(out_root)
        except Exception as e:
            print(f"(cleanup warning) failed to rm {out_root}: {e}")


if __name__ == "__main__":
    main()
