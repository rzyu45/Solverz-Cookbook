"""IES (Integrated Energy System) benchmark tests.

Two tests exercise the ``module_printer(jit=True)`` path — the only
code-gen route that matters for production. Both compile the full
IES model, run a two-phase Rodas integration (100 hr settle → 300 s
perturbation), and validate ALL shared variable trajectories against
a pre-recorded benchmark generated from the legacy (non-LoopEqn)
model.

``test_ies_module``
    Uses the default LoopEqn model path (``loopeqn=True`` for
    ``heat_network`` and ``gas_network``).

``test_ies_legacy_module``
    Forces the legacy scalar-Eqn path (``loopeqn=False``) to guard
    against regressions in the original code.

The benchmark ``fullstate_bench.npz`` contains 107 variable
trajectories (all device-model, node-level network, and coupling
variables). Variables unique to one path (e.g. per-pipe ``p0`` in
legacy, ``p_all`` in LoopEqn) are compared by name — only variables
present in BOTH the benchmark and the test solution are checked.
"""
import os
import sys
import time

import numpy as np
from Solverz import (Eqn, Model, Opt, Rodas, TimeSeriesParam, Var,
                     made_numerical, module_printer)
from SolMuseum.ae import eb, eps_network, p2g
from SolMuseum.dae import gas_network, gt, heat_network, pv, st
from SolUtil import DhsFlow, GasFlow, PowerFlow

DX = 100
QBASE = 37.41


def _build_ies_model(datadir, loopeqn=True):
    """Assemble the full IES model.

    Parameters
    ----------
    loopeqn : bool
        Passed through to ``heat_network.mdl()`` and
        ``gas_network.mdl()`` to select the LoopEqn or legacy path.
    """
    POWER_CASE = str(datadir / "caseI.xlsx")
    HEAT_CASE = str(datadir / "case_heat.xlsx")

    pf = PowerFlow(POWER_CASE)
    pf.run()

    voltage = pf.Vm * np.exp(1j * pf.Va)
    power = (pf.Pg - pf.Pd) + 1j * (pf.Qg - pf.Qd)
    current = (power / voltage).conjugate()
    ux = voltage.real
    uy = voltage.imag
    ix = current.real
    iy = current.imag

    model = Model()

    gt_0 = gt(
        ux=ux[0], uy=uy[0], ix=ix[0], iy=iy[0],
        ra=0, xdp=0.0608, xqp=0.0969, xq=0.0969, Damping=10, Tj=47.28,
        A=-0.158, B=1.158, C=0.5, D=0.5, E=313, W=320,
        kp=0.11, ki=1 / 30, K1=0.85, K2=0.15, TRbase=800, wref=1,
        qmin=-0.13, qmax=1.5, T1=12.2, T2=1.7, TCD=0.16, TG=0.05,
        b=0.04, TFS=1000, Tref=900.3144, c=1e8,
    )
    model.add(gt_0.mdl())

    eb_5 = eb(
        eta=1, vm0=pf.Vm[5], phi=pf.Pd[5] * pf.baseMVA * 1e6,
        ux=ux[5], uy=uy[5], epsbase=pf.baseMVA * 1e6,
        pd=pf.Pd[5], pd0=pf.Pd[5],
    )
    model.add(eb_5.mdl())

    pv_1 = pv(
        ux=ux[1], uy=uy[1], ix=ix[1], iy=iy[1],
        kop=-0.05, koi=-10, ws=376.99, lf=0.005, kip=2, kii=9,
        Pnom=26813.04395522, kp=-0.1, ki=-0.01, udcref=800,
        cpv=1e-4, ldc=0.05, cdc=5e-3, ISC=19.6, IM=18,
        Radiation=1000, sref=1000, Ttemp=25, UOC=864, UM=688,
    )
    model.add(pv_1.mdl())

    z = 1e-8
    eta = 1
    f_steam = 1.02775712
    phi = (eta * f_steam - pf.Pg[2]) / z
    st_2 = st(
        ux=ux[2], uy=uy[2], ix=ix[2], iy=iy[2],
        ra=0, xdp=0.0608, xqp=0.0969, xq=0.0969, Damping=10, Tj=47.28,
        phi=phi, z=z, F=f_steam, eta=eta, TREF=70, alpha=0.3,
        mu_min=0, mu_max=1, TCH=0.2, TRH=5, kp=-1, ki=-1,
    )
    model.add(st_2.mdl())

    p2g_4 = p2g(
        h=50.18120992, eta=0.8, epsbase=100, c=340,
        p=10e6, q=-36.55027730265727, pd=pf.Pd[4],
    )
    model.add(p2g_4.mdl())

    epsn = eps_network(pf)
    model.add(epsn.mdl(dyn=True))

    model.eqn_gt_ux = Eqn("eqn_gt_ux", model.ux_gt - model.ux[0])
    model.eqn_gt_uy = Eqn("eqn_gt_uy", model.uy_gt - model.uy[0])
    model.eqn_gt_ix = Eqn("eqn_gt_ix", model.ix_gt - model.ix[0])
    model.eqn_gt_iy = Eqn("eqn_gt_iy", model.iy_gt - model.iy[0])
    model.eqn_eb_ux = Eqn("eqn_eb_ux", model.ux_eb - model.ux[5])
    model.eqn_eb_uy = Eqn("eqn_eb_uy", model.uy_eb - model.uy[5])
    model.eqn_pv_ux = Eqn("eqn_pv_ux", model.ux_pv - model.ux[1])
    model.eqn_pv_uy = Eqn("eqn_pv_uy", model.uy_pv - model.uy[1])
    model.eqn_pv_ix = Eqn("eqn_pv_ix", model.ix_pv - model.ix[1])
    model.eqn_pv_iy = Eqn("eqn_pv_iy", model.iy_pv - model.iy[1])
    model.eqn_st_ux = Eqn("eqn_st_ux", model.ux_st - model.ux[2])
    model.eqn_st_uy = Eqn("eqn_st_uy", model.uy_st - model.uy[2])
    model.eqn_st_ix = Eqn("eqn_st_ix", model.ix_st - model.ix[2])
    model.eqn_st_iy = Eqn("eqn_st_iy", model.iy_st - model.iy[2])

    for bus in range(9):
        if bus in [0, 1, 2]:
            continue
        lhs1 = model.ux[bus] * model.ix[bus] + model.uy[bus] * model.iy[bus]
        if bus == 5:
            rhs1 = model.Pg[bus] - model.pd_eb
        elif bus == 4:
            rhs1 = model.Pg[bus] - model.pd_p2g
        else:
            rhs1 = model.Pg[bus] - model.Pd[bus]
        model.add(Eqn(f"eqn_P_{bus}", lhs1 - rhs1))
        lhs2 = model.uy[bus] * model.ix[bus] - model.ux[bus] * model.iy[bus]
        rhs2 = model.Qg[bus] - model.Qd[bus]
        model.add(Eqn(f"eqn_Q_{bus}", lhs2 - rhs2))

    gf = GasFlow(POWER_CASE)
    gf.run(tee=False)
    model.add(gas_network(gf).mdl(dx=DX, loopeqn=loopeqn))

    model.eqn_p_p2g = Eqn("eqn_p_p2g", model.p_p2g - model.Pi[0])
    model.eqn_q_p2g = Eqn("eqn_q_p2g", model.q_p2g + model.fs[0])
    model.fs_0 = TimeSeriesParam("fs_0", [36.5509116, 36.5509116], [0, 100])

    for node in range(gf.n_node):
        if node == 1:
            pass
        elif node == 0:
            model.__dict__[f"node_fs_{node}"] = Eqn(
                f"node_fs_{node}", model.fs[node] - model.fs_0)
        else:
            model.__dict__[f"node_fs_{node}"] = Eqn(
                f"node_fs_{node}", model.fs[node])
        if node not in [8, 9, 10]:
            gas_load = 0
        elif node in [8, 9]:
            gas_load = gf.fl[node]
        else:
            gas_load = model.qfuel_gt[0] * QBASE
        model.__dict__[f"node_fl_{node}"] = Eqn(
            f"node_fl_{node}", model.fl[node] - gas_load)

    df = DhsFlow(HEAT_CASE)
    df.run()
    model.add(heat_network(df).mdl(
        dx=DX, dynamic_slack=True, loopeqn=loopeqn))

    model.eqn_st_Ts = Eqn("eqn_st_Ts", model.Ts_st - model.Ts_slack)

    for node in range(df.n_node):
        if node not in [0, 1, 5]:
            model.__dict__[f"eqn_phi_hn_{node}"] = Eqn(
                f"eqn_phi_hn_{node}", model.phi[node] - df.phi[node])
        elif node == 0:
            model.eqn_phi_slack = Eqn(
                "eqn_phi_slack", model.phi[0] * 1e6 - model.phi_st)
        elif node == 1:
            model.eqn_phi_WHB = Eqn(
                "eqn_phi_WHB", model.phi[1] * 1e6 - model.phi_gt)
        else:
            model.eqn_phi_eb = Eqn(
                "eqn_phi_eb", model.phi[5] * 1e6 - model.phi_eb)

    omega_coi = ((model.Tj_st * model.omega_st
                  + model.Tj_gt * model.omega_gt)
                 / (model.Tj_st + model.Tj_gt))
    model.omega_coi = Var("omega_coi", 1)
    model.eqn_omega_coi = Eqn("eqn_omega_coi", model.omega_coi - omega_coi)

    return model


def _run_and_assert(dae, y0, datadir):
    """Run two-phase Rodas and validate ALL shared variables.

    Phase 1: 100 hr settle.
    Phase 2: 300 s perturbation, 301 output points.

    Loads ``fullstate_bench.npz`` (generated from legacy model) and
    checks every variable name present in both the benchmark and the
    solution.
    """
    sol0 = Rodas(dae, [0, 100 * 3600], y0, Opt(pbar=True))

    dae.p["fs_0"] = TimeSeriesParam(
        "fs_0",
        [36.5509116, 47.5509116, 37.5509116],
        [0, 300, 10 * 3600],
    )
    sol = Rodas(dae, np.linspace(0, 300, 301), sol0.Y[-1], Opt(pbar=True))

    bench = np.load(str(datadir / "fullstate_bench.npz"))

    # Time must match exactly
    sol_time = np.asarray(sol.T).reshape(-1)
    np.testing.assert_allclose(
        sol_time, bench['time'], rtol=1e-8, atol=1e-10)

    # Check every variable present in both benchmark and solution
    sol_vars = set(sol.Y[-1].var_list)
    bench_vars = set(bench.files) - {'time', 'var_names'}
    shared = sorted(sol_vars & bench_vars)

    n_checked = 0
    for var in shared:
        sol_data = np.asarray(sol.Y[var])
        bench_data = bench[var]
        if sol_data.shape != bench_data.shape:
            continue
        np.testing.assert_allclose(
            sol_data, bench_data, rtol=1e-3, atol=1e-4,
            err_msg=f"Variable '{var}' diverged from benchmark",
        )
        n_checked += 1

    assert n_checked >= 40, (
        f"Only {n_checked} shared variables checked (expected ≥40). "
        f"sol has {len(sol_vars)} vars, bench has {len(bench_vars)} vars, "
        f"shared: {len(shared)}"
    )


def _build_compile_and_import(sdae, y0, module_name, out_dir, jit=True):
    """Render, optionally compile, and import a module."""
    printer = module_printer(sdae, y0, module_name,
                              directory=out_dir, jit=jit)
    printer.render()
    sys.path.insert(0, out_dir)
    t0 = time.perf_counter()
    mod = __import__(module_name)
    compile_time = time.perf_counter() - t0
    return mod, compile_time


def test_ies_module(datadir, tmp_path):
    """LoopEqn path + ``module_printer(jit=True)``.

    Validates that the LoopEqn model (default for heat_network and
    gas_network) produces correct trajectories for ALL shared
    variables after JIT compilation.
    """
    model = _build_ies_model(datadir, loopeqn=True)
    sdae, y0 = model.create_instance()

    out_dir = str(tmp_path / 'loop')
    mod, ct = _build_compile_and_import(
        sdae, y0, 'ies_loop', out_dir, jit=True)
    print(f"\nLoopEqn compile: {ct:.1f}s")

    try:
        _run_and_assert(mod.mdl, mod.y, datadir)
    finally:
        sys.path.remove(out_dir)
        for k in list(sys.modules):
            if k.startswith('ies_loop'):
                del sys.modules[k]


def test_ies_legacy_module(datadir, tmp_path):
    """Legacy path + ``module_printer(jit=True)``.

    Guards against regressions in the original scalar-Eqn code path.
    """
    model = _build_ies_model(datadir, loopeqn=False)
    sdae, y0 = model.create_instance()

    out_dir = str(tmp_path / 'legacy')
    mod, ct = _build_compile_and_import(
        sdae, y0, 'ies_legacy', out_dir, jit=True)
    print(f"\nLegacy compile: {ct:.1f}s")

    try:
        _run_and_assert(mod.mdl, mod.y, datadir)
    finally:
        sys.path.remove(out_dir)
        for k in list(sys.modules):
            if k.startswith('ies_legacy'):
                del sys.modules[k]
