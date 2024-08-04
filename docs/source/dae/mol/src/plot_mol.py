import numpy as np
import pandas as pd
import os
import networkx as nx

from Solverz import load, Eqn, Ode, Abs, Model, Var, Param, TimeSeriesParam, made_numerical, Rodas, Opt
import matplotlib.pyplot as plt

# %%
# create G of the network using networkx
idx_pipe = np.arange(10)
idx_node = np.arange(11)
idx_from = [0, 3, 1, 4, 4, 5, 7, 7, 2, 6]
idx_to = [2, 4, 5, 8, 6, 6, 9, 10, 3, 7]

G = nx.DiGraph()

for i in range(len(idx_pipe)):
    G.add_node(idx_from[i])
    G.add_node(idx_to[i])
    G.add_edge(idx_from[i], idx_to[i], idx=idx_pipe[i])
A = nx.incidence_matrix(G,
                        nodelist=idx_node,
                        edgelist=sorted(G.edges(data=True), key=lambda edge: edge[2].get('idx', 1)),
                        oriented=True)

# %%
m = Model()
Piv = np.array([10., 8., 9.25356807, 8.44138875, 7.54225206,
                7.67010339, 7.32536498, 6.09107411, 7.27354882, 5.56663968,
                5.87805022]) * 1e6
m.Pi = Var('Pi', value=Piv)  # node pressure
f = np.array([39.57682738, 39.57682738, 23.73641444, 20.83, 18.74682738,
              23.73641444, 25.81324182, 16.67, 39.57682738, 42.48324182])
m.qn = Var('qn', value=A @ f)  # node injection mass flow
va = 340
m.D = Param('D', value=0.5 * np.ones(10))
m.Area = Param('Area', np.pi * (m.D.value / 2) ** 2)
m.lam = Param('lam', value=0.03 * np.ones(10))
Piset0 = 10e6
m.Piset0 = TimeSeriesParam('Piset0',
                           v_series=[Piset0, Piset0 + 0.5e6, Piset0 + 0.5e6],
                           time_series=[0, 2 * 3600, 10 * 3600])
m.Piset1 = Param('Piset1', value=8e6)
L = 51000*np.ones(10)
dx = 100
M = np.floor(L / dx).astype(int)
for j in range(10):
    p0 = np.linspace(Piv[idx_from[j]], Piv[idx_to[j]], M[j] + 1)
    m.__dict__['p' + str(j)] = Var('p' + str(j), value=p0)
    m.__dict__['q' + str(j)] = Var('q' + str(j), value=f[j] * np.ones(M[j] + 1))

# % method of lines
for node in G.nodes:
    eqn_q = m.qn[node]
    for edge in G.in_edges(node, data=True):
        pipe = edge[2]['idx']
        idx = str(pipe)
        qi = m.__dict__['q' + idx]
        pi = m.__dict__['p' + idx]
        eqn_q = eqn_q - qi[M[pipe]]
        m.__dict__[f'pressure_outlet_pipe{idx}'] = Eqn(f'Pressure node {node} pipe {idx} outlet',
                                                       m.Pi[node] - pi[M[pipe]])

    for edge in G.out_edges(node, data=True):
        pipe = edge[2]['idx']
        idx = str(pipe)
        qi = m.__dict__['q' + idx]
        pi = m.__dict__['p' + idx]
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
                             m.qn[8] - 20.83)
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
for edge in G.edges(data=True):
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

# %% initialize m
sdae, y0 = m.create_instance()
ndae, code = made_numerical(sdae, y0, sparse=True, output_code=True)

sol = Rodas(ndae,
            [0, 3600 * 50],
            y0)

# %% visualize
plt.plot(sol.T / 3600, sol.Y['Pi'], label=[f'P at N{i}' for i in range(11)])
plt.xlim([0, 50])
plt.xlabel('Time/s')
plt.ylabel(r'Pressure/Mpa')
plt.legend(ncols=3)
plt.grid()
plt.show()
