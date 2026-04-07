"""
Integrated energy system simulation example for Solverz Cookbook.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Solverz import Eqn, Model, Opt, Rodas, TimeSeriesParam, Var, made_numerical
from SolMuseum.ae import eb, eps_network, p2g
from SolMuseum.dae import gas_network, gt, heat_network, pv, st
from SolUtil import DhsFlow, GasFlow, PowerFlow


DATA_DIR = Path(__file__).resolve().parent / "test_ies"
POWER_CASE = DATA_DIR / "caseI.xlsx"
HEAT_CASE = DATA_DIR / "case_heat.xlsx"
DX = 100
QBASE = 37.41


def initialize_power_flow():
    pf = PowerFlow(str(POWER_CASE))
    pf.run()

    u = pf.Vm * np.exp(1j * pf.Va)
    s = (pf.Pg - pf.Pd) + 1j * (pf.Qg - pf.Qd)
    i = (s / u).conjugate()
    return pf, u.real, u.imag, i.real, i.imag


def add_power_subsystems(model, pf, ux, uy, ix, iy):
    gt_0 = gt(
        ux=ux[0],
        uy=uy[0],
        ix=ix[0],
        iy=iy[0],
        ra=0,
        xdp=0.0608,
        xqp=0.0969,
        xq=0.0969,
        Damping=10,
        Tj=47.28,
        A=-0.158,
        B=1.158,
        C=0.5,
        D=0.5,
        E=313,
        W=320,
        kp=0.11,
        ki=1 / 30,
        K1=0.85,
        K2=0.15,
        TRbase=800,
        wref=1,
        qmin=-0.13,
        qmax=1.5,
        T1=12.2,
        T2=1.7,
        TCD=0.16,
        TG=0.05,
        b=0.04,
        TFS=1000,
        Tref=900.3144,
        c=1e8,
    )
    model.add(gt_0.mdl())

    eb_5 = eb(
        eta=1,
        vm0=pf.Vm[5],
        phi=pf.Pd[5] * pf.baseMVA * 1e6,
        ux=ux[5],
        uy=uy[5],
        epsbase=pf.baseMVA * 1e6,
        pd=pf.Pd[5],
        pd0=pf.Pd[5],
    )
    model.add(eb_5.mdl())

    pv_1 = pv(
        ux=ux[1],
        uy=uy[1],
        ix=ix[1],
        iy=iy[1],
        kop=-0.05,
        koi=-10,
        ws=376.99,
        lf=0.005,
        kip=2,
        kii=9,
        Pnom=26813.04395522,
        kp=-0.1,
        ki=-0.01,
        udcref=800,
        cpv=1e-4,
        ldc=0.05,
        cdc=5e-3,
        ISC=19.6,
        IM=18,
        Radiation=1000,
        sref=1000,
        Ttemp=25,
        UOC=864,
        UM=688,
    )
    model.add(pv_1.mdl())

    z = 1e-8
    eta = 1
    f_steam = 1.02775712
    phi = (eta * f_steam - pf.Pg[2]) / z
    st_2 = st(
        ux=ux[2],
        uy=uy[2],
        ix=ix[2],
        iy=iy[2],
        ra=0,
        xdp=0.0608,
        xqp=0.0969,
        xq=0.0969,
        Damping=10,
        Tj=47.28,
        phi=phi,
        z=z,
        F=f_steam,
        eta=eta,
        TREF=70,
        alpha=0.3,
        mu_min=0,
        mu_max=1,
        TCH=0.2,
        TRH=5,
        kp=-1,
        ki=-1,
    )
    model.add(st_2.mdl())

    p2g_4 = p2g(
        h=50.18120992,
        eta=0.8,
        epsbase=100,
        c=340,
        p=10e6,
        q=-36.55027730265727,
        pd=pf.Pd[4],
    )
    model.add(p2g_4.mdl())

    model.add(eps_network(pf).mdl(dyn=True))

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

        active_lhs = model.ux[bus] * model.ix[bus] + model.uy[bus] * model.iy[bus]
        if bus == 5:
            active_rhs = model.Pg[bus] - model.pd_eb
        elif bus == 4:
            active_rhs = model.Pg[bus] - model.pd_p2g
        else:
            active_rhs = model.Pg[bus] - model.Pd[bus]
        model.add(Eqn(f"eqn_P_{bus}", active_lhs - active_rhs))

        reactive_lhs = model.uy[bus] * model.ix[bus] - model.ux[bus] * model.iy[bus]
        reactive_rhs = model.Qg[bus] - model.Qd[bus]
        model.add(Eqn(f"eqn_Q_{bus}", reactive_lhs - reactive_rhs))


def add_gas_subsystem(model):
    gf = GasFlow(str(POWER_CASE))
    gf.run(tee=False)

    model.add(gas_network(gf).mdl(dx=DX))
    model.eqn_p_p2g = Eqn("eqn_p_p2g", model.p_p2g - model.Pi[0])
    model.eqn_q_p2g = Eqn("eqn_q_p2g", model.q_p2g + model.fs[0])
    model.fs_0 = TimeSeriesParam("fs_0", [36.5509116, 36.5509116], [0, 100])

    for node in range(gf.n_node):
        if node == 1:
            pass
        elif node == 0:
            model.__dict__[f"node_fs_{node}"] = Eqn(f"node_fs_{node}", model.fs[node] - model.fs_0)
        else:
            model.__dict__[f"node_fs_{node}"] = Eqn(f"node_fs_{node}", model.fs[node])

        if node not in [8, 9, 10]:
            load_rhs = 0
        elif node in [8, 9]:
            load_rhs = gf.fl[node]
        else:
            load_rhs = model.qfuel_gt[0] * QBASE

        model.__dict__[f"node_fl_{node}"] = Eqn(f"node_fl_{node}", model.fl[node] - load_rhs)


def add_heat_subsystem(model):
    df = DhsFlow(str(HEAT_CASE))
    df.run()

    model.add(heat_network(df).mdl(dx=DX, dynamic_slack=True))
    model.eqn_st_Ts = Eqn("eqn_st_Ts", model.Ts_st - model.Ts_slack)

    for node in range(df.n_node):
        if node not in [0, 1, 5]:
            model.__dict__[f"eqn_phi_hn_{node}"] = Eqn(f"eqn_phi_hn_{node}", model.phi[node] - df.phi[node])
        elif node == 0:
            model.eqn_phi_slack = Eqn("eqn_phi_slack", model.phi[0] * 1e6 - model.phi_st)
        elif node == 1:
            model.eqn_phi_WHB = Eqn("eqn_phi_WHB", model.phi[1] * 1e6 - model.phi_gt)
        else:
            model.eqn_phi_eb = Eqn("eqn_phi_eb", model.phi[5] * 1e6 - model.phi_eb)


def build_model():
    pf, ux, uy, ix, iy = initialize_power_flow()

    model = Model()
    add_power_subsystems(model, pf, ux, uy, ix, iy)
    add_gas_subsystem(model)
    add_heat_subsystem(model)

    omega_coi = (model.Tj_st * model.omega_st + model.Tj_gt * model.omega_gt) / (model.Tj_st + model.Tj_gt)
    model.omega_coi = Var("omega_coi", 1)
    model.eqn_omega_coi = Eqn("eqn_omega_coi", model.omega_coi - omega_coi)
    return model


def solve_steady_state():
    sdae, y0 = build_model().create_instance()
    dae = made_numerical(sdae, y0, sparse=True)
    sol0 = Rodas(dae, [0, 100 * 3600], y0, Opt(pbar=True))
    return dae, sol0


def run_disturbance():
    dae, sol0 = solve_steady_state()
    dae.p["fs_0"] = TimeSeriesParam(
        "fs_0",
        [36.5509116, 37.5509116, 37.5509116],
        [0, 5, 6],
    )
    short_term = Rodas(dae, [0, 300], sol0.Y[-1], Opt(pbar=True))
    long_term = Rodas(dae, [300, 10 * 3600], short_term.Y[-1], Opt(pbar=True))
    return short_term, long_term


def plot_frequency_response():
    short_term, long_term = run_disturbance()

    time = np.concatenate([short_term.T, long_term.T[1:]])
    omega_gt = np.concatenate([short_term.Y["omega_gt"], long_term.Y["omega_gt"][1:]])
    omega_st = np.concatenate([short_term.Y["omega_st"], long_term.Y["omega_st"][1:]])
    omega_coi = np.concatenate([short_term.Y["omega_coi"], long_term.Y["omega_coi"][1:]])

    plt.plot(time, omega_gt, label="GT")
    plt.plot(time, omega_st, label="ST")
    plt.plot(time, omega_coi, label="COI")
    plt.xlabel("Time / s")
    plt.ylabel("Frequency / p.u.")
    plt.title("Frequency response after gas-source disturbance")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_frequency_response()
