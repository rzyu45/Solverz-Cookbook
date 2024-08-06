import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import numpy as np

from Solverz import Var, Eqn, Model, sin, cos, Param, module_printer

current_dir = os.getcwd()
file_path = os.path.join(current_dir, "test_pf_jac", "pf.mat")
sys = loadmat(file_path)
file_path = os.path.join(current_dir, "test_pf_jac", "pq.mat")
PQ = loadmat(file_path)
# %% model
V = sys["V"].reshape((-1,))
nb = V.shape[0]
Ybus = sys["Ybus"]
G = Ybus.real
B = Ybus.imag
ref = (sys["ref"] - 1).reshape((-1,)).tolist()
pv = (sys["pv"] - 1).reshape((-1,)).tolist()
pq = (sys["pq"] - 1).reshape((-1,)).tolist()
npv = len(pv)
npq = len(pq)
mbase = 100  # MVA

m = Model()
m.Va = Var("Va", np.angle(V)[pv + pq])
m.Vm = Var("Vm", np.abs(V)[pq])
m.Pg = Param("Pg", PQ["Pg"].reshape(-1, ) / mbase)
m.Qg = Param("Qg", PQ["Qg"].reshape(-1, ) / mbase)
m.Pd = Param("Pd", PQ["Pd"].reshape(-1, ) / mbase)
m.Qd = Param("Qd", PQ["Qd"].reshape(-1, ) / mbase)


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
        expr += Vmi * Vmj * (G[i, j] * cos(Vai - Vaj) + B[i, j] * sin(Vai - Vaj))
    m.__dict__[f"P_eqn_{i}"] = Eqn(f"P_eqn_{i}", expr + m.Pd[i] - m.Pg[i])

for i in pq:
    expr = 0
    Vmi = get_Vm(i)
    Vai = get_Va(i)
    for j in range(nb):
        Vmj = get_Vm(j)
        Vaj = get_Va(j)
        expr += Vmi * Vmj * (G[i, j] * sin(Vai - Vaj) - B[i, j] * cos(Vai - Vaj))
    m.__dict__[f"Q_eqn_{i}"] = Eqn(f"Q_eqn_{i}", expr + m.Qd[i] - m.Qg[i])
# %% create instance
spf, y0 = m.create_instance()
pyprinter = module_printer(spf, y0, "powerflow", make_hvp=True, jit=True)
pyprinter_njit = module_printer(spf, y0, "powerflow_njit", make_hvp=True)
pyprinter.render()
pyprinter_njit.render()
