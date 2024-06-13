import numpy as np
import pandas as pd
import os

from Solverz import load, Eqn, Ode, Abs, Model, Var, Param, TimeSeriesParam, made_numerical, Rodas, Opt
import matplotlib.pyplot as plt


# %%
def test_mol(datadir):
    stdy_mdl = load(datadir/'steady_gas_flow_model.pkl')

    mpc = stdy_mdl['mpc']
    gc = stdy_mdl['gc']
    gtc = stdy_mdl['gtc']
    p2gc = stdy_mdl['p2gc']

    graph = gc['G']
    mdl_ss = stdy_mdl['model']
    p = mdl_ss.p
    y_ss = stdy_mdl['var']
    idx_rfrom = gc['rpipe_from']
    idx_rto = gc['rpipe_to']
    idx_rp = gc['idx_rpipe']
    # %%
    m = Model()
    Piv = y_ss['Pi'] * 1e6
    m.Pi = Var('Pi', value=Piv)  # node pressure
    m.qn = Var('qn', value=gc['A'] @ y_ss['f'])  # node injection mass flow
    va = gc['va']
    m.D = Param('D', value=gc['D'])
    m.Area = Param('Area', np.pi * (gc['D'] / 2) ** 2)
    m.lam = Param('lam', value=gc['lam'])
    Piset0 = p['Piset'][0] * 1e6
    m.Piset0 = TimeSeriesParam('Piset0',
                               v_series=[Piset0, Piset0 + 0.5e6, Piset0 + 0.5e6],
                               time_series=[0, 2 * 3600, 10 * 3600])
    m.Piset1 = Param('Piset1', value=p['Piset'][1] * 1e6)
    L = gc['L']
    dx = 100
    M = np.floor(L / dx).astype(int)
    for j in range(gc['n_pipe']):
        p0 = np.linspace(Piv[idx_rfrom[j]], Piv[idx_rto[j]], M[j] + 1)
        m.__dict__['p' + str(j)] = Var('p' + str(j), value=p0)
        m.__dict__['q' + str(j)] = Var('q' + str(j), value=y_ss['f'][j] * np.ones(M[j] + 1))

    # % method of lines
    for node in graph.nodes:
        eqn_q = m.qn[node]
        for edge in graph.in_edges(node, data=True):
            pipe = edge[2]['idx']
            idx = str(pipe)
            ptype = edge[2]['type']
            qi = m.__dict__['q' + idx]
            pi = m.__dict__['p' + idx]
            if ptype == 1:
                eqn_q = eqn_q - qi[M[pipe]]
                m.__dict__[f'pressure_outlet_pipe{idx}'] = Eqn(f'Pressure node {node} pipe {idx} outlet',
                                                               m.Pi[node] - pi[M[pipe]])

        for edge in graph.out_edges(node, data=True):
            pipe = edge[2]['idx']
            idx = str(pipe)
            ptype = edge[2]['type']
            qi = m.__dict__['q' + idx]
            pi = m.__dict__['p' + idx]
            if ptype == 1:
                eqn_q = eqn_q + qi[0]
                m.__dict__[f'pressure_inlet_pipe{idx}'] = Eqn(f'Pressure node {node} pipe {idx} inlet',
                                                              m.Pi[node] - pi[0])

        m.__dict__[f'mass_continuity_node{node}'] = Eqn('mass flow continuity of node {}'.format(node), eqn_q)

    m.Pressure_source_node0 = Eqn('Pressure of source node0',
                                  m.Pi[0] - m.Piset0)
    m.Pressure_source_node1 = Eqn('Pressure of source node1',
                                  m.Pi[1] - m.Piset1)
    m.mass_injection_ns_nodes = Eqn('Mass flow injection of non-source node',
                                    m.qn[2:8])
    m.mass_injection_node8 = Eqn('Mass flow injection of node 8',
                                 m.qn[8] - gc['finset'][-3])
    m.mass_injection_node9 = Eqn('Mass flow injection of node 9',
                                 m.qn[9] - 18.81898528)
    m.mass_injection_node10 = Eqn('Mass flow injection of node 10',
                                  m.qn[10] - 15.12876112)

    def mol_tvd1_p_eqn_rhs1(P_list, Q_list, S, va, dx):
        P_list = [arg.symbol if isinstance(arg, Var) else arg for arg in P_list]
        Q_list = [arg.symbol if isinstance(arg, Var) else arg for arg in Q_list]
        pm1, p0, pp1 = P_list
        qm1, q0, qp1 = Q_list
        # return -va ** 2 / S * (qp1 - qm1) / (2 * dx) + va * (pp1 - 2 * p0 + pm1) / (2 * dx)
        return -va ** 2 / S * (qp1 - qm1) / (2 * dx)

    def mol_tvd1_q_eqn_rhs1(P_list, Q_list, S, va, lam, D, dx):
        P_list = [arg.symbol if isinstance(arg, Var) else arg for arg in P_list]
        Q_list = [arg.symbol if isinstance(arg, Var) else arg for arg in Q_list]
        pm1, p0, pp1 = P_list
        qm1, q0, qp1 = Q_list
        # return -S * (pp1 - pm1) / (2 * dx) + va * (qp1 - 2 * q0 + qm1) / (2 * dx) - lam * va ** 2 * q0 * Abs(q0) / (
        #         2 * D * S * p0)
        return -S * (pp1 - pm1) / (2 * dx) - lam * va ** 2 * q0 * Abs(q0) / (
                2 * D * S * p0)

    # method of lines
    p_list = []
    q_list = []
    for edge in graph.edges(data=True):
        f_node = edge[0]
        t_node = edge[1]
        j = edge[2]['idx']
        Mj = M[j]
        pj = m.__dict__['p' + str(j)]
        qj = m.__dict__['q' + str(j)]
        Dj = m.D[j]
        Sj = m.Area[j]
        lamj = m.lam[j]

        rhs = mol_tvd1_q_eqn_rhs1([pj[0:Mj - 1], pj[1:Mj], pj[2:Mj + 1]],
                                  [qj[0:Mj - 1], qj[1:Mj], qj[2:Mj + 1]],
                                  Sj,
                                  va,
                                  lamj,
                                  Dj,
                                  dx)
        m.__dict__['q' + str(j) + '_eqn2'] = Ode(f'weno3-q{j}2',
                                                 rhs,
                                                 qj[1:Mj])

        rhs = mol_tvd1_p_eqn_rhs1([pj[0:Mj - 1], pj[1:Mj], pj[2:Mj + 1]],
                                  [qj[0:Mj - 1], qj[1:Mj], qj[2:Mj + 1]],
                                  Sj,
                                  va,
                                  dx)
        m.__dict__['p' + str(j) + '_eqn3'] = Ode(f'weno3-p{j}3',
                                                 rhs,
                                                 pj[1:Mj])

        m.__dict__['p' + str(j) + 'bd1'] = Eqn(pj.name + 'bd1',
                                               Sj * pj[Mj] + va * qj[Mj] + Sj * pj[Mj - 2] + va * qj[
                                                   Mj - 2] - 2 * (
                                                       Sj * pj[Mj - 1] + va * qj[Mj - 1]))
        m.__dict__['q' + str(j) + 'bd2'] = Eqn(qj.name + 'bd2',
                                               Sj * pj[2] - va * qj[2] + Sj * pj[0] - va * qj[0] - 2 * (
                                                       Sj * pj[1] - va * qj[1]))

    # %% create instance
    sdae, y0 = m.create_instance()
    ndae, code = made_numerical(sdae, y0, sparse=True, output_code=True)

    sol = Rodas(ndae,
                [0, 3600 * 50],
                y0)

    # %% test
    with open(datadir/'P_bench.npy', 'rb') as f:
        Pibench = np.load(f)
    np.testing.assert_allclose(sol.Y['Pi'], Pibench)
