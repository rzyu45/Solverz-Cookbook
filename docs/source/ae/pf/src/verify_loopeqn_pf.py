"""Verify that loopeqn=False and loopeqn=True produce the same NR
solution on the case30 power-flow problem.

Uses ``made_numerical`` (jit=False) so both builds are cheap; the
point here is *correctness*, not compile-time.
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Re-use the mock PowerFlow builder from the benchmark script.
from bench_loopeqn_pf import _load_mock_pf  # noqa: E402

from SolMuseum.ae import eps_network  # noqa: E402
from Solverz import Model, made_numerical, nr_method  # noqa: E402

warnings.filterwarnings("ignore")


def build_and_solve(pf, loopeqn):
    # eps_network(pf).mdl() returns a Model. Wrap it: the cookbook
    # polar PF path needs pv/pq pins — that's what _add_pq_pins adds
    # in the benchmark, so we mirror that here.
    from bench_loopeqn_pf import _add_pq_pins
    epsn = eps_network(pf)
    m = epsn.mdl(dyn=False, loopeqn=loopeqn)
    _add_pq_pins(m, pf, loopeqn)
    spf, y0 = m.create_instance()
    mdl = made_numerical(spf, y0, sparse=True)
    # Perturb slightly so NR actually iterates.  Mutate the underlying
    # flat array (Vars doesn't expose a public copy — we want a fresh
    # one anyway).
    y_run = type(y0)(y0.a, y0.array.copy())
    if 'Vm' in y_run.var_list:
        y_run['Vm'] = y_run['Vm'] + 0.05
    if 'Va' in y_run.var_list:
        y_run['Va'] = y_run['Va'] + 0.02
    sol = nr_method(mdl, y_run)
    return sol, y0


def main():
    pf = _load_mock_pf()
    print("Building + solving loopeqn=False...")
    sol_f, y0_f = build_and_solve(pf, loopeqn=False)
    print(f"  nstep={sol_f.stats.nstep}  succeed={sol_f.stats.succeed}")

    print("Building + solving loopeqn=True...")
    sol_t, y0_t = build_and_solve(pf, loopeqn=True)
    print(f"  nstep={sol_t.stats.nstep}  succeed={sol_t.stats.succeed}")

    # Compare the complex bus voltage V[i] = Vm[i] * exp(1j * Va[i]) at
    # shared buses. The two paths store different subsets:
    #   loopeqn=False: Va[pv+pq] and Vm[pq] (reduced state)
    #   loopeqn=True:  Va[:] and Vm[:] full-length (ref/pv pinned via
    #                  LoopEqn pin equations)
    # Align them by bus index and compare.
    print("\n=== Solution comparison (complex voltage per bus) ===")
    ref = list(pf.idx_slack)
    pv = list(pf.idx_pv)
    pq = list(pf.idx_pq)
    pv_pq = list(pv) + list(pq)

    # loopeqn=False reduced vectors
    Vm_f_red = np.asarray(sol_f.y['Vm']).ravel()
    Va_f_red = np.asarray(sol_f.y['Va']).ravel()
    # loopeqn=True full vectors
    Vm_t_full = np.asarray(sol_t.y['Vm']).ravel()
    Va_t_full = np.asarray(sol_t.y['Va']).ravel()

    # Reconstruct full V for loopeqn=False using pinned ref/pv.
    nb = len(Vm_t_full)
    Vm_f_full = pf.Vm.copy()
    Va_f_full = pf.Va.copy()
    for idx, bus in enumerate(pv_pq):
        Va_f_full[bus] = Va_f_red[idx]
    for idx, bus in enumerate(pq):
        Vm_f_full[bus] = Vm_f_red[idx]

    V_f = Vm_f_full * np.exp(1j * Va_f_full)
    V_t = Vm_t_full * np.exp(1j * Va_t_full)

    max_abs = np.max(np.abs(V_f - V_t))
    max_rel = max_abs / (np.max(np.abs(V_f)) + 1e-15)
    print(f"  buses: {nb} (ref={ref}, #pv={len(pv)}, #pq={len(pq)})")
    print(f"  max|V_f - V_t| = {max_abs:.3e}  (relative {max_rel:.3e})")
    if max_rel > 1e-6:
        print("  WARNING: complex-voltage relative error exceeds 1e-6")
    else:
        print("  OK: both paths agree to 1e-6 relative tolerance")


if __name__ == '__main__':
    main()
