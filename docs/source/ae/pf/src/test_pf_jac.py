import os
import sys
import shutil

import numpy as np
from Solverz import Var, Eqn, Model, sin, cos, made_numerical, Param, module_printer, sicnm, Opt
from scipy.io import loadmat


def test_pf_jac(datadir):
    mat = loadmat(datadir / 'pf.mat')
    V = mat["V"].reshape((-1,))
    nb = V.shape[0]
    Ybus = mat["Ybus"]
    G = Ybus.real.toarray()
    B = Ybus.imag.toarray()
    Jbench = mat["J2"]
    Hzbench = mat["Hz"]
    v0 = mat["z0"]
    ref = (mat["ref"] - 1).reshape((-1,)).tolist()
    pv = (mat["pv"] - 1).reshape((-1,)).tolist()
    pq = (mat["pq"] - 1).reshape((-1,)).tolist()

    m = Model()
    m.Va = Var("Va", np.angle(V)[pv + pq])
    m.Vm = Var("Vm", np.abs(V)[pq])

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
        Vmi = get_Vm(i)
        Vai = get_Va(i)
        for j in range(nb):
            Vmj = get_Vm(j)
            Vaj = get_Va(j)
            expr += Vmi * Vmj * (G[i, j] * cos(Vai - Vaj) +
                                 B[i, j] * sin(Vai - Vaj))
        m.__dict__[f"P_eqn_{i}"] = Eqn(f"P_eqn_{i}", expr)

    for i in pq:
        expr = 0
        Vmi = get_Vm(i)
        Vai = get_Va(i)
        for j in range(nb):
            Vmj = get_Vm(j)
            Vaj = get_Va(j)
            expr += Vmi * Vmj * (G[i, j] * sin(Vai - Vaj) -
                                 B[i, j] * cos(Vai - Vaj))
        m.__dict__[f"Q_eqn_{i}"] = Eqn(f"Q_eqn_{i}", expr)

    spf, y0 = m.create_instance()
    pf, code = made_numerical(spf,
                              y0,
                              sparse=True,
                              make_hvp=True,
                              output_code=True)
    j = pf.J(y0, pf.p)
    np.testing.assert_allclose(j.tocoo().data, Jbench.tocoo().data, atol=1e-8, rtol=1e-7)
    Hz = pf.HVP(y0, pf.p, v0)
    dH = Hz - Hzbench
    np.testing.assert_almost_equal(dH.tocoo().data.max(), 0)


def test_pf(datadir):
    param = loadmat(datadir / 'pf.mat')
    PQ = loadmat(datadir / 'pq.mat')
    # %% model
    V = param["V"].reshape((-1,))
    nb = V.shape[0]
    Ybus = param["Ybus"]
    G = Ybus.real.toarray()
    B = Ybus.imag.toarray()
    ref = (param["ref"] - 1).reshape((-1,)).tolist()
    pv = (param["pv"] - 1).reshape((-1,)).tolist()
    pq = (param["pq"] - 1).reshape((-1,)).tolist()

    m = Model()
    m.Va = Var("Va", np.angle(V)[pv + pq])
    m.Vm = Var("Vm", np.abs(V)[pq])
    m.Pg = Param('Pg', PQ['Pg'].reshape(-1, ) / 100)
    m.Qg = Param('Qg', PQ['Qg'].reshape(-1, ) / 100)
    m.Pd = Param('Pd', PQ['Pd'].reshape(-1, ) / 100)
    m.Qd = Param('Qd', PQ['Qd'].reshape(-1, ) / 100)

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
        Vmi = get_Vm(i)
        Vai = get_Va(i)
        for j in range(nb):
            Vmj = get_Vm(j)
            Vaj = get_Va(j)
            expr += Vmi * Vmj * (G[i, j] * cos(Vai - Vaj) +
                                 B[i, j] * sin(Vai - Vaj))
        m.__dict__[f"P_eqn_{i}"] = Eqn(f"P_eqn_{i}", expr + m.Pd[i] - m.Pg[i])

    for i in pq:
        expr = 0
        Vmi = get_Vm(i)
        Vai = get_Va(i)
        for j in range(nb):
            Vmj = get_Vm(j)
            Vaj = get_Va(j)
            expr += Vmi * Vmj * (G[i, j] * sin(Vai - Vaj) -
                                 B[i, j] * cos(Vai - Vaj))
        m.__dict__[f"Q_eqn_{i}"] = Eqn(f"Q_eqn_{i}", expr + m.Qd[i] - m.Qg[i])
    # %% create instance
    spf, y0 = m.create_instance()

    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)

    test_folder_path = current_folder + '\\Solverz_cookbook_ae'
    sys.path.extend([test_folder_path])

    pyprinter_njit = module_printer(spf,
                                    y0,
                                    'power_flow',
                                    directory=test_folder_path,
                                    make_hvp=True)
    pyprinter_njit.render()

    mat = loadmat(datadir / 'ill-Va.mat')

    pv = (mat['pv'] - 1).reshape(-1).tolist()
    pq = (mat['pq'] - 1).reshape(-1).tolist()
    Va0 = np.deg2rad(mat['Va0'])

    for i in range(len(pv + pq)):
        y0['Va'][i:i + 1] = Va0[(pv + pq)[i]]

    from power_flow import mdl as pf, y as y0

    sol1 = sicnm(pf, y0, Opt(scheme='rodas3d', rtol=1e-1, atol=1e-1, hinit=0.1, ite_tol=1e-5))

    Va_bench = np.array([-0.00725168, 0.02576391, -0.05921429, -0.02773726, -0.01445899,
                         -0.0265652, -0.03132391, -0.03252984, -0.03956586, -0.04628328,
                         -0.04757365, -0.05230635, -0.05890374, -0.05230635, -0.02682417,
                         -0.04028282, -0.04034914, -0.04615499, -0.05920749, -0.06070932,
                         -0.0690837, -0.06756212, -0.06088395, -0.04592767, -0.02949587,
                         -0.03733863, -0.03954792, -0.0371493, -0.0530846])
    Vm_bench = np.array([0.98313825, 0.98009295, 0.98240617, 0.97318397, 0.9673554,
                         0.96062366, 0.98050609, 0.98440428, 0.98050609, 0.9854683,
                         0.97667682, 0.98022901, 0.97739562, 0.97686538, 0.96844031,
                         0.96528702, 0.96916633, 0.99338328, 0.9885663, 0.99021484,
                         0.97219415, 0.97471485, 0.9795967, 0.96788288])
    np.testing.assert_allclose(sol1.y['Vm'], Vm_bench, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sol1.y['Va'], Va_bench, rtol=1e-6, atol=1e-6)

    shutil.rmtree(test_folder_path)
