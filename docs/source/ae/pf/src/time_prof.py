"""Compilation cost and per-call cost of a jit module against a non-jit one.

Run ``python pf_mdl.py`` first: it renders ``powerflow`` with ``jit=True``
and ``powerflow_njit`` without. Then run this script. It writes
``../fig/time_prof.png`` and ``../fig/time_prof_01.png``, the two figures
the chapter shows, and prints the numbers the prose quotes.

The compilation figure is the wall time of the first import of the jit
module, which is where Numba compiles every kernel. The per-call figure is
a steady-state median, not a first call: a first call to the jit module
would still be compiling, and a first call to either would also be paying
the page faults of a cold module.
"""
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent.parent / 'fig'
WARMUP, REPEATS = 10, 2000

# Font sizes shared by both figures so the chapter's two images match.
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'legend.fontsize': 10})


def cold_import_seconds(module: str) -> float:
    """Time the FIRST import of ``module`` in a fresh interpreter.

    It has to be a subprocess: within one process the module is imported
    once and the Numba cache is warm from then on, so a second measurement
    in the same process would report the cache hit rather than the compile.
    """
    code = (f"import time; t = time.perf_counter(); import {module}; "
            f"print(time.perf_counter() - t)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, cwd=Path(__file__).resolve().parent)
    if out.returncode != 0:
        raise SystemExit(f"importing {module} failed:\n{out.stdout}\n{out.stderr}")
    return float(out.stdout.strip().splitlines()[-1])


def per_call(fn, *args) -> float:
    for _ in range(WARMUP):
        fn(*args)
    samples = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t)
    return statistics.median(samples)


compilation_time = cold_import_seconds('powerflow')

from powerflow import mdl as pf, y as y0            # noqa: E402
from powerflow_njit import mdl as pf1, y as y1      # noqa: E402

Fjit, Jjit = per_call(pf.F, y0, pf.p), per_call(pf.J, y0, pf.p)
Fnjit, Jnjit = per_call(pf1.F, y1, pf1.p), per_call(pf1.J, y1, pf1.p)

print(f"cold import of the jit module : {compilation_time:.1f} s")
print(f"F  jit {Fjit*1e6:9.2f} us | non-jit {Fnjit*1e6:9.2f} us "
      f"| ratio {Fnjit/Fjit:.1f}")
print(f"J  jit {Jjit*1e6:9.2f} us | non-jit {Jnjit*1e6:9.2f} us "
      f"| ratio {Jnjit/Jjit:.1f}")

FIG.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(4, 3.2))
ax.bar([0], [compilation_time], width=0.4, color='tab:green')
ax.set_xticks([0], ['Compilation time'])
ax.set_xlim(-0.5, 0.5)
top = 20 * (int(compilation_time // 20) + 1)
ax.set_ylim(0, top)
ax.set_yticks(range(0, top + 1, 20))
ax.set_ylabel('Time/s')
ax.grid(axis='y')
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG / 'time_prof.png', dpi=200)

fig, ax = plt.subplots(figsize=(4, 3.2))
ax.scatter([1], [Fjit], c='tab:orange', label='F eval')
ax.scatter([2], [Fnjit], c='tab:orange')
ax.scatter([1], [Jjit], c='tab:purple', label='J eval')
ax.scatter([2], [Jnjit], c='tab:purple')
ax.set_yscale('log')
ax.set_ylim(1e-6, 1e-2)
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2], ['jit', 'Non-jit'])
ax.set_ylabel('Time/s')
ax.legend()
ax.grid()
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG / 'time_prof_01.png', dpi=200)
print(f"figures written to {FIG}")
