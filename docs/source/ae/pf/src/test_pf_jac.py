import os
import sys
import shutil

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_array
from Solverz import (Var, Eqn, Model, sin, cos, made_numerical, Param,
                     module_printer, sicnm, Opt, Mat_Mul, nr_method)


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


def test_pf_matmul(datadir):
    """Cross-validate Mat_Mul (rectangular) vs element-wise (polar) power flow.

    Both formulations solve the same case30 system from a flat start.
    The solutions are converted to the same representation and compared.
    """
    param = loadmat(datadir / 'pf.mat')
    PQ = loadmat(datadir / 'pq.mat')

    V = param["V"].reshape((-1,))
    nb = V.shape[0]
    Ybus = param["Ybus"].tocsc()
    G_full = Ybus.real
    B_full = Ybus.imag
    G_dense = G_full.toarray()
    B_dense = B_full.toarray()
    ref = (param["ref"] - 1).reshape((-1,)).tolist()
    pv = (param["pv"] - 1).reshape((-1,)).tolist()
    pq = (param["pq"] - 1).reshape((-1,)).tolist()
    non_ref = pv + pq
    mbase = 100
    e0 = V.real
    f0 = V.imag
    Pg = PQ["Pg"].reshape(-1) / mbase
    Qg = PQ["Qg"].reshape(-1) / mbase
    Pd = PQ["Pd"].reshape(-1) / mbase
    Qd = PQ["Qd"].reshape(-1) / mbase
    Pinj = Pg - Pd
    Qinj = Qg - Qd

    # --- 1. Element-wise polar form ---
    m1 = Model()
    m1.Va = Var("Va", np.zeros(len(non_ref)))
    m1.Vm = Var("Vm", np.ones(len(pq)))
    m1.Pg = Param('Pg', Pg)
    m1.Qg = Param('Qg', Qg)
    m1.Pd = Param('Pd', Pd)
    m1.Qd = Param('Qd', Qd)

    def get_Vm(idx):
        if idx in ref + pv:
            return np.abs(V)[idx]
        elif idx in pq:
            return m1.Vm[pq.index(idx)]

    def get_Va(idx):
        if idx in ref:
            return np.angle(V)[idx]
        elif idx in pv + pq:
            return m1.Va[(pv + pq).index(idx)]

    for i in pv + pq:
        expr = 0
        Vmi = get_Vm(i)
        Vai = get_Va(i)
        for j in range(nb):
            Vmj = get_Vm(j)
            Vaj = get_Va(j)
            expr += Vmi * Vmj * (G_dense[i, j] * cos(Vai - Vaj)
                                 + B_dense[i, j] * sin(Vai - Vaj))
        m1.__dict__[f"P_eqn_{i}"] = Eqn(f"P_eqn_{i}", expr + m1.Pd[i] - m1.Pg[i])

    for i in pq:
        expr = 0
        Vmi = get_Vm(i)
        Vai = get_Va(i)
        for j in range(nb):
            Vmj = get_Vm(j)
            Vaj = get_Va(j)
            expr += Vmi * Vmj * (G_dense[i, j] * sin(Vai - Vaj)
                                 - B_dense[i, j] * cos(Vai - Vaj))
        m1.__dict__[f"Q_eqn_{i}"] = Eqn(f"Q_eqn_{i}", expr + m1.Qd[i] - m1.Qg[i])

    spf1, y01 = m1.create_instance()
    mdl1 = made_numerical(spf1, y01, sparse=True)
    sol1 = nr_method(mdl1, y01)

    # Reconstruct polar solution as full voltage
    Vm_polar = np.abs(V).copy()
    Va_polar = np.angle(V).copy()
    Va_polar[pv + pq] = sol1.y["Va"]
    Vm_polar[pq] = sol1.y["Vm"]

    # --- 2. Mat_Mul rectangular form ---
    n_nr = len(non_ref)
    n_pq = len(pq)
    n_pv = len(pv)

    G_nr = csc_array(G_full[np.ix_(non_ref, non_ref)])
    B_nr = csc_array(B_full[np.ix_(non_ref, non_ref)])
    e_ref = e0[ref[0]]
    f_ref = f0[ref[0]]
    G_ref_col = G_full[non_ref, ref[0]].toarray().ravel()
    B_ref_col = B_full[non_ref, ref[0]].toarray().ravel()
    p_ref = G_ref_col * e_ref - B_ref_col * f_ref
    q_ref = B_ref_col * e_ref + G_ref_col * f_ref

    pq_in_nr = [non_ref.index(i) for i in pq]
    G_pq = csc_array(G_full[np.ix_(pq, non_ref)])
    B_pq = csc_array(B_full[np.ix_(pq, non_ref)])
    G_pq_ref_col = G_full[pq, ref[0]].toarray().ravel()
    B_pq_ref_col = B_full[pq, ref[0]].toarray().ravel()
    p_ref_pq = G_pq_ref_col * e_ref - B_pq_ref_col * f_ref
    q_ref_pq = B_pq_ref_col * e_ref + G_pq_ref_col * f_ref

    m2 = Model()
    m2.e = Var("e", np.ones(n_nr))
    m2.f = Var("f", np.zeros(n_nr))
    m2.G_nr = Param("G_nr", G_nr, dim=2, sparse=True)
    m2.B_nr = Param("B_nr", B_nr, dim=2, sparse=True)
    m2.G_pq = Param("G_pq", G_pq, dim=2, sparse=True)
    m2.B_pq = Param("B_pq", B_pq, dim=2, sparse=True)
    m2.p_ref = Param("p_ref", p_ref)
    m2.q_ref = Param("q_ref", q_ref)
    m2.p_ref_pq = Param("p_ref_pq", p_ref_pq)
    m2.q_ref_pq = Param("q_ref_pq", q_ref_pq)
    m2.Pinj = Param("Pinj", Pinj[non_ref])
    m2.Qinj = Param("Qinj", Qinj[pq])

    m2.P_eqn = Eqn("P_balance",
                    m2.e * (Mat_Mul(m2.G_nr, m2.e) - Mat_Mul(m2.B_nr, m2.f) + m2.p_ref)
                    + m2.f * (Mat_Mul(m2.B_nr, m2.e) + Mat_Mul(m2.G_nr, m2.f) + m2.q_ref)
                    - m2.Pinj)

    e_pq = m2.e[pq_in_nr[0]:pq_in_nr[-1] + 1]
    f_pq = m2.f[pq_in_nr[0]:pq_in_nr[-1] + 1]
    m2.Q_eqn = Eqn("Q_balance",
                    f_pq * (Mat_Mul(m2.G_pq, m2.e) - Mat_Mul(m2.B_pq, m2.f) + m2.p_ref_pq)
                    - e_pq * (Mat_Mul(m2.B_pq, m2.e) + Mat_Mul(m2.G_pq, m2.f) + m2.q_ref_pq)
                    - m2.Qinj)

    pv_in_nr = [non_ref.index(i) for i in pv]
    Vm_pv_sq = np.abs(V[pv]) ** 2
    m2.Vm_sq = Param("Vm_sq", Vm_pv_sq)
    e_pv = m2.e[pv_in_nr[0]:pv_in_nr[-1] + 1]
    f_pv = m2.f[pv_in_nr[0]:pv_in_nr[-1] + 1]
    m2.V_eqn = Eqn("V_pv", e_pv ** 2 + f_pv ** 2 - m2.Vm_sq)

    spf2, y02 = m2.create_instance()
    mdl2 = made_numerical(spf2, y02, sparse=True)
    sol2 = nr_method(mdl2, y02)

    # Reconstruct rectangular solution as full voltage
    e_sol = np.zeros(nb)
    f_sol = np.zeros(nb)
    e_sol[ref] = e_ref
    f_sol[ref] = f_ref
    e_sol[non_ref] = sol2.y["e"]
    f_sol[non_ref] = sol2.y["f"]
    Vm_rect = np.sqrt(e_sol ** 2 + f_sol ** 2)
    Va_rect = np.arctan2(f_sol, e_sol)

    # --- 3. Cross-validate ---
    np.testing.assert_allclose(Vm_rect[non_ref], Vm_polar[non_ref], atol=1e-5,
                               err_msg="Vm mismatch between Mat_Mul and element-wise forms")
    np.testing.assert_allclose(Va_rect[non_ref], Va_polar[non_ref], atol=1e-4,
                               err_msg="Va mismatch between Mat_Mul and element-wise forms")
