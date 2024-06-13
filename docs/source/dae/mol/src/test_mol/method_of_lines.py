import numpy as np

from Code.eps_function import load_mpc, load_mac, plus_load_impedance
from Code.ngs_function import mol_tvd1_p_eqn_rhs1, mol_tvd1_q_eqn_rhs1
from Code.tf2ss import tf2ss
from Solverz import (load, Eqn, Ode, Abs, cos, sin, TimeSeriesParam, Model, Var, Param, Min, Saturation,
                     AntiWindUp, made_numerical, Rodas, Opt)

# %%
stdy_mdl = load('steady_gas_flow_model.pkl')

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
mpc = load_mpc('caseI.xlsx')
Pd = mpc['Pd']
Qd = mpc['Qd']
Pg = mpc['Pg']
Qg = mpc['Qg']
Vm = mpc['Vm']
Va = mpc['Va']
Ybus = plus_load_impedance(mpc['Ybus'], np.zeros_like(Qd), Qd, Vm)
Gbus = Ybus.real
Bbus = Ybus.imag
mac = load_mac('caseI.xlsx')
# %%
m = Model()
Piv = y_ss['Pi'] * 1e6
m.Pi = Var('Pi', value=Piv)  # node pressure
m.qn = Var('qn', value=gc['A'] @ y_ss['f'])  # node injection mass flow
va = gc['va']
m.D = Param('D', value=gc['D'])
m.Area = Param('Area', np.pi * (gc['D'] / 2) ** 2)
m.lam = Param('lam', value=gc['lam'])
m.Piset = Param('Piset', value=p['Piset'] * 1e6)
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

m.__dict__['Pressure_source_node'] = Eqn('Pressure of source node',
                                         m.Pi[0:2] - m.Piset)
m.__dict__['mass_injection_ns_nodes'] = Eqn('Mass flow injection of non-source node',
                                            m.qn[2:8])
m.__dict__['mass_injection_node8'] = Eqn('Mass flow injection of node 8',
                                         m.qn[8] - gc['finset'][-3])

# finite difference and initialize variables
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

    # rhs = -Sj * (pj[1:Mj + 1] - pj[0:Mj]) / dx - lamj * va ** 2 * qj[0:Mj] * Abs(qj[0:Mj]) / (
    #             2 * Dj * Sj * pj[0:Mj])
    # m.__dict__['q' + str(j) + '_eqn'] = Ode(f'q{j}_eqn',
    #                                         f=rhs,
    #                                         diff_var=qj[0:Mj])
    # 
    # m.__dict__['p' + str(j) + '_eqn'] = Ode(f'p{j}_eqn',
    #                                         -va ** 2 / Sj * (qj[1:Mj + 1] - qj[0:Mj]) / dx,
    #                                         pj[1:Mj + 1])

    rhs = mol_tvd1_q_eqn_rhs1([pj[0:Mj-1], pj[1:Mj], pj[2:Mj+1]],
                              [qj[0:Mj-1], qj[1:Mj], qj[2:Mj+1]],
                              Sj,
                              va,
                              lamj,
                              Dj,
                              dx)
    m.__dict__['q' + str(j) + '_eqn2'] = Ode(f'weno3-q{j}2',
                                             rhs,
                                             qj[1:Mj])

    rhs = mol_tvd1_p_eqn_rhs1([pj[0:Mj-1], pj[1:Mj], pj[2:Mj+1]],
                              [qj[0:Mj-1], qj[1:Mj], qj[2:Mj+1]],
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

# %% gt
m.TG = Param('TG', 0.05 * np.ones(3))
m.Fmax = Param('Fmax', 1.5 * np.ones(3))
m.Fmin = Param('Fmin', -0.13 * np.ones(3))
m.kNL = Param('kNL', 0.24 * np.ones(3))
m.b = Param('b', 0.04 * np.ones(3))
m.TFS = Param('TFS', 600 * np.ones(3))
m.TCD = Param('TCD', 0.16 * np.ones(3))
m.A = Param('A', -0.158 * np.ones(3))
m.B = Param('B', 1.158 * np.ones(3))
m.C = Param('C', 0.5 * np.ones(3))
m.D1 = Param('D1', -413 * np.ones(3))
m.E = Param('E', 313 * np.ones(3))
GSH = [0.85, 0.85, 0.85]
TSH = [12.2, 12.2, 12.2]
m.TTR = Param('TTR', 1.7 * np.ones(3))
m.TR = Param('TR', 602 * np.ones(3))
m.GTC = Param('GTC', 3.3 * np.ones(3))
m.TTC = Param('TTC', 30 * np.ones(3))
m.Damp = Param('Damp', 10 * np.ones(3))
m.Pl6 = Param('Pl6', Pd[5])
m.Pl8 = Param('Pl8', Pd[7])
m.Xdp = Param('Xdp', value=mac['xdp'])
m.Xd = Param('Xd', value=mac['xd'])
m.Xqp = Param('Xqp', value=mac['xq'])
m.Xq = Param('Xq', value=mac['xq'])
m.Tj = Param('Tj', value=mac['Tj'])
m.Damp = Param('Damp', value=mac['D'])
m.ra = Param('ra', value=mac['ra'])
U = Vm * (np.cos(Va) + 1j * np.sin(Va))
S = Pg - Pd + 1j * (Qg - Qd)
I = (S / U).conj()
EQ = U[0:3] + (m.ra.value + 1j * m.Xq.value) * I[0:3]
diat_n = np.angle(EQ)
w_n = np.ones((3,))
Ud = np.sin(diat_n) * U[0:3].real - np.cos(diat_n) * U[0:3].imag
Uq = np.cos(diat_n) * U[0:3].real + np.sin(diat_n) * U[0:3].imag
Id = np.sin(diat_n) * I[0:3].real - np.cos(diat_n) * I[0:3].imag
Iq = np.cos(diat_n) * I[0:3].real + np.sin(diat_n) * I[0:3].imag
m.Eqp = Param('Eqp', value=Uq + m.Xdp.value * Id + m.ra.value * Iq)
m.Edp = Param('Edp', value=np.zeros((3,)))
wb = 2 * np.pi * 60

m.Ix = Var('Ix', I.real)
m.Iy = Var('Iy', I.imag)
m.Ux = Var('Ux', U.real)
m.Uy = Var('Uy', U.imag)
m.delta = Var('delta', diat_n)
m.omega = Var('omega', w_n)

omega_coi = (m.Tj[0] * m.omega[0] + m.Tj[1] * m.omega[1] + m.Tj[2] * m.omega[2]) / (
        m.Tj[0] + m.Tj[1] + m.Tj[2])
for i in range(3):
    m.__dict__[f'delta_eq{i}'] = Ode(f'Delta equation{i}',
                                     wb * (m.omega[i] - omega_coi),
                                     diff_var=m.delta[i])
Pe_mac = m.Ux[0:3] * m.Ix[0:3] + m.Uy[0:3] * m.Iy[0:3] + (m.Ix[0:3] ** 2 + m.Iy[0:3] ** 2) * m.ra
# Exhaust temperature
m.Te = Var('Te', value=[512.4688739, 512.4688739, 512.4688739])
m.mf = Var('mf', value=[0.7618, 0.7618, 0.7618])
m.TRbase = Param('TRbase', value=[602., 602., 602.])
m.ExhTemp = Eqn('Exhaust Temperature', m.Te - (m.TRbase + m.D1 * (1 - m.mf) + m.E * (1 - m.omega)))
# Radiation shield
RS_A_v = np.zeros((3,))
RS_B_v = np.zeros((3,))
RS_C_v = np.zeros((3,))
RS_D_v = np.zeros((3,))
for i in range(3):
    RS = tf2ss([GSH[i] * TSH[i], 1], [TSH[i], 1])
    RS_A_v[i] = RS.A
    RS_B_v[i] = RS.B
    RS_C_v[i] = RS.C
    RS_D_v[i] = RS.D
m.RS_A = Param('RS_A', RS_A_v)
m.RS_B = Param('RS_B', RS_B_v)
m.RS_C = Param('RS_C', RS_C_v)
m.RS_D = Param('RS_D', RS_D_v)

m.Tr = Var('Tr', value=[512.5, 512.5, 512.5])
m.Tr_i = Var('Tr_i', init=(m.Tr - m.RS_D * m.Te) / m.RS_C)
m.RS1 = Ode('Radiation Shield1', f=m.RS_A * m.Tr_i + m.RS_B * m.Te, diff_var=m.Tr_i)
m.RS2 = Eqn('Radiation Shield2', m.Tr - m.RS_C * m.Tr_i - m.RS_D * m.Te)
# Thermocouple
m.Tx = Var('Tx', value=[512.5, 512.5, 512.5])
m.TC = Ode('Thermocouple', f=(m.Tr - m.Tx) / m.TTR, diff_var=m.Tx)
# Thermal controller
kp = m.GTC / m.TTC
ki = 1 / m.TTC
m.qT_i = Var('qT_i', value=[0, 0, 0])
m.qT = Var('qT', init=ki * m.qT_i + kp * (m.TR - m.Tx))
m.TC1 = Ode('Temp controller1', f=AntiWindUp(m.qT, m.Fmin, m.Fmax, m.TR - m.Tx), diff_var=m.qT_i)
m.TC2 = Eqn('Temp controller2', m.qT - (ki * m.qT_i + kp * (m.TR - m.Tx)))
# Compressor
m.Cop = Var('Cop', value=[0.7618, 0.7618, 0.7618])
m.CompDisc = Ode('Compressor discharge', (m.mf - m.Cop) / m.TCD, diff_var=m.Cop)
# Tmec
m.Tmec = Var('Tmec', value=[0.7383, 0.7383, 0.7383])
m.TorqMec = Eqn('Torque mechanic', m.Tmec - (m.A + m.B * m.Cop + m.C * (1 - m.omega)))
# speed governor
m.qR = Var('qR', value=[0, 0, 0])
m.W = Param('W', [35, 50, 15])
m.SpeedGov = Ode('Speed governor', f=(m.W * (1 - m.omega) - m.qR) / m.TG, diff_var=m.qR)
m.qfuel = Var('qfuel',
              init=(m.kNL + (1 - m.kNL) * Min(m.qT, Saturation(m.qR, m.Fmin, m.Fmax)) * m.omega))
rhs = m.qfuel - (m.kNL + (1 - m.kNL) * Min(m.qT, Saturation(m.qR, m.Fmin, m.Fmax)) * m.omega)
m.FuelCons = Eqn('Fuel Consumption', rhs)
# Valve Positioner
m.xv = Var('xv', value=[0.7617, 0.7617, 0.7617])
m.ValPos = Ode('Valve Positioner', (m.qfuel - m.xv) / m.b, diff_var=m.xv)
m.FuelDyn = Ode('Fuel Dynamics', (m.xv - m.mf) / m.TFS, diff_var=m.mf)
m.omega_eq = Ode('Omega equation',
                 (m.Tmec * m.omega - Pe_mac - m.Damp * (m.omega - 1)) / m.Tj,
                 diff_var=m.omega)
m.Edp_eq = Eqn('Edp equation',
               m.Edp - sin(m.delta) * (m.Ux[0:3] + m.ra * m.Ix[0:3] - m.Xqp * m.Iy[0:3])
               + cos(m.delta) * (m.Uy[0:3] + m.ra * m.Iy[0:3] + m.Xqp * m.Ix[0:3]))
m.Eqp_eq = Eqn('Eqp equation',
               m.Eqp - cos(m.delta) * (m.Ux[0:3] + m.ra * m.Ix[0:3] - m.Xdp * m.Iy[0:3])
               - sin(m.delta) * (m.Uy[0:3] + m.ra * m.Iy[0:3] + m.Xdp * m.Ix[0:3]))
m.Pl5 = Var('Pl5', 0.2870)

for i in range(9):
    rhs1 = m.Ix[i]
    for j in range(9):
        rhs1 = rhs1 - Gbus[i, j] * m.Ux[j] + Bbus[i, j] * m.Uy[j]
    m.__dict__[f'Ix_inj_{i}'] = Eqn(f'Ix injection {i}', rhs1)

for i in range(9):
    rhs2 = m.Iy[i]
    for j in range(9):
        rhs2 = rhs2 - Gbus[i, j] * m.Uy[j] - Bbus[i, j] * m.Ux[j]
    m.__dict__[f'Iy_inj_{i}'] = Eqn(f'Iy injection {i}', rhs2)

for i in range(3, 9):
    rhs = m.Ux[i] * m.Ix[i] + m.Uy[i] * m.Iy[i]
    if i == 4:
        rhs = rhs + m.Pl5
    elif i == 5:
        rhs = rhs + m.Pl6
    elif i == 7:
        rhs = rhs + m.Pl8
    m.__dict__[f'Pl_eq{i}'] = Eqn(f'Active load eqn{i}',
                                  rhs)

m.Ql_eq = Eqn('Reactive load eqn',
              m.Uy[3:9] * m.Ix[3:9] - m.Ux[3:9] * m.Iy[3:9])
m.hg_p2g = Param('hg_p2g', p2gc['hg'][0])
m.eta_p2g = Param('eta_p2g', p2gc['npg'][0])
m.eqnp2g = Eqn('EQN P2G',
               m.Pl5 - Abs(m.qn[0]) * 340 ** 2 * m.hg_p2g / (m.eta_p2g * m.Pi[0]) / 100)
qb = 37.41
m.mass_injection_node10 = Eqn('Mass flow injection of node 10',
                              m.qn[10] - m.qfuel[0] * qb)
m.mass_injection_node9 = Eqn('Mass flow injection of node 9',
                             m.qn[9] - m.qfuel[1] * qb)
# %%
sdae, y0 = m.create_instance()
# % initialize m
ndae, code = made_numerical(sdae, y0, sparse=True, output_code=True)

y0['qfuel'] = np.array([0.40440429, 0.50304692, 0.30576172])
y0['Ix'] = np.array([4.11577801e-001, 5.57289748e-001, 2.92411518e-001,
                     0, -2.98934848e-001, 0,
                     0, 0, 0])
y0['Iy'] = np.array([-4.53995606e-001, -2.87680380e-001, -1.99377440e-002,
                     0, -6.50121790e-003, 0,
                     0, 0, 0])
y0['Ux'] = np.array([1.02577211, 0.96525758, 0.98162259, 0.99962196, 0.95962155,
                     0.97948724, 0.94727755, 0.94530532, 0.98045424])
y0['Uy'] = np.array([0.05060987, 0.08860008, 0.05317543, 0.02690298, 0.02086979,
                     0.00289433, 0.05376947, 0.02831728, 0.03604012])
y0['delta'] = np.array([0.08438947, 0.43923751, 0.39607036])
y0['omega'] = np.array([1, 1, 1])

ndae.p['TFS'] = np.array([1000., 1000., 1000.])
ndae.p['W'] = np.array([320., 400., 360.])
ndae.p['Pl6'] = np.array([0.3])
ndae.p['Pl8'] = np.array([0.3])
ndae.p['TRbase'] = np.array([800., 800., 800.])

sol = Rodas(ndae,
            [0, 3600],
            y0,
            Opt(rtol=1e-1, atol=1e-3))
