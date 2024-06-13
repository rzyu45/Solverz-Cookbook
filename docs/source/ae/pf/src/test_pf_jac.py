from scipy.io import loadmat
import os
import numpy as np

from Solverz import Var, Eqn, Model, sin, cos, made_numerical


def test_pf_jac(datadir):
    mat = loadmat(datadir / 'pf.mat')
    V = mat["V"].reshape((-1,))
    nb = V.shape[0]
    Ybus = mat["Ybus"]
    G = Ybus.real
    B = Ybus.imag
    Jbench = mat["J2"]
    Hzbench = mat["Hz"]
    v0 = mat["z0"]
    ref = (mat["ref"] - 1).reshape((-1,)).tolist()
    pv = (mat["pv"] - 1).reshape((-1,)).tolist()
    pq = (mat["pq"] - 1).reshape((-1,)).tolist()
    npv = len(pv)
    npq = len(pq)

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
                              output_code=True)
    j = pf.J(y0, pf.p)
    np.testing.assert_allclose(j.tocoo().data, Jbench.tocoo().data, atol=1e-8, rtol=1e-7)
