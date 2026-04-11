"""
Power flow modelling using Mat_Mul (rectangular coordinates).

Instead of element-wise for-loops, this formulation uses Mat_Mul to express
the power injection equations in compact matrix-vector form:

    P = e * (G@e - B@f) + f * (B@e + G@f)
    Q = f * (G@e - B@f) - e * (B@e + G@f)

where G and B are the conductance and susceptance matrices (sparse),
and e, f are the real and imaginary parts of the bus voltage.

Solverz automatically computes the symbolic Jacobian via matrix calculus.
"""
from scipy.io import loadmat
from scipy.sparse import csc_array
import os
import numpy as np

from Solverz import Var, Eqn, Model, Param, Mat_Mul, made_numerical, nr_method

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "test_pf_jac", "pf.mat")
sys_data = loadmat(file_path)
file_path = os.path.join(current_dir, "test_pf_jac", "pq.mat")
PQ = loadmat(file_path)

# %% parse system data
V = sys_data["V"].reshape((-1,))
nb = V.shape[0]
Ybus = sys_data["Ybus"].tocsc()
G_full = Ybus.real
B_full = Ybus.imag
ref = (sys_data["ref"] - 1).reshape((-1,)).tolist()
pv = (sys_data["pv"] - 1).reshape((-1,)).tolist()
pq = (sys_data["pq"] - 1).reshape((-1,)).tolist()
non_ref = pv + pq
mbase = 100

# Initial voltage in rectangular coordinates
e0 = V.real
f0 = V.imag
Pg = PQ["Pg"].reshape(-1) / mbase
Qg = PQ["Qg"].reshape(-1) / mbase
Pd = PQ["Pd"].reshape(-1) / mbase
Qd = PQ["Qd"].reshape(-1) / mbase
Pinj = Pg - Pd
Qinj = Qg - Qd

# %% Submatrices for non-reference buses
#
# P equations:  P_i = e_i * sum_j(G_ij*e_j - B_ij*f_j)
#                   + f_i * sum_j(B_ij*e_j + G_ij*f_j)
#
# In matrix form (only non-ref buses):
#   P = e * (G_nr@e - B_nr@f + p_ref) + f * (B_nr@e + G_nr@f + q_ref)
#
# where p_ref, q_ref absorb the reference bus contributions.

n_nr = len(non_ref)
n_pq = len(pq)
n_pv = len(pv)

G_nr = csc_array(G_full[np.ix_(non_ref, non_ref)])
B_nr = csc_array(B_full[np.ix_(non_ref, non_ref)])

# Reference bus contribution (constant)
e_ref = e0[ref[0]]
f_ref = f0[ref[0]]
G_ref_col = G_full[non_ref, ref[0]].toarray().ravel()
B_ref_col = B_full[non_ref, ref[0]].toarray().ravel()
p_ref = G_ref_col * e_ref - B_ref_col * f_ref
q_ref = B_ref_col * e_ref + G_ref_col * f_ref

# %% Q equations use a separate submatrix (PQ rows only)
pq_in_nr = [non_ref.index(i) for i in pq]

G_pq = csc_array(G_full[np.ix_(pq, non_ref)])
B_pq = csc_array(B_full[np.ix_(pq, non_ref)])

G_pq_ref_col = G_full[pq, ref[0]].toarray().ravel()
B_pq_ref_col = B_full[pq, ref[0]].toarray().ravel()
p_ref_pq = G_pq_ref_col * e_ref - B_pq_ref_col * f_ref
q_ref_pq = B_pq_ref_col * e_ref + G_pq_ref_col * f_ref

# %% Build model with Mat_Mul
m = Model()

# Variables: e and f at non-reference buses (flat start)
m.e = Var("e", np.ones(n_nr))
m.f = Var("f", np.zeros(n_nr))

# Parameters
m.G_nr = Param("G_nr", G_nr, dim=2, sparse=True)
m.B_nr = Param("B_nr", B_nr, dim=2, sparse=True)
m.G_pq = Param("G_pq", G_pq, dim=2, sparse=True)
m.B_pq = Param("B_pq", B_pq, dim=2, sparse=True)
m.p_ref = Param("p_ref", p_ref)
m.q_ref = Param("q_ref", q_ref)
m.p_ref_pq = Param("p_ref_pq", p_ref_pq)
m.q_ref_pq = Param("q_ref_pq", q_ref_pq)
m.Pinj = Param("Pinj", Pinj[non_ref])
m.Qinj = Param("Qinj", Qinj[pq])

# Active power balance (all non-ref buses): n_nr equations
m.P_eqn = Eqn("P_balance",
               m.e * (Mat_Mul(m.G_nr, m.e) - Mat_Mul(m.B_nr, m.f) + m.p_ref)
               + m.f * (Mat_Mul(m.B_nr, m.e) + Mat_Mul(m.G_nr, m.f) + m.q_ref)
               - m.Pinj)

# Reactive power balance (PQ buses only): n_pq equations
# Uses G_pq/B_pq submatrices (rows = pq buses, cols = non-ref buses)
e_pq = m.e[pq_in_nr[0]:pq_in_nr[-1] + 1]
f_pq = m.f[pq_in_nr[0]:pq_in_nr[-1] + 1]
m.Q_eqn = Eqn("Q_balance",
               f_pq * (Mat_Mul(m.G_pq, m.e) - Mat_Mul(m.B_pq, m.f) + m.p_ref_pq)
               - e_pq * (Mat_Mul(m.B_pq, m.e) + Mat_Mul(m.G_pq, m.f) + m.q_ref_pq)
               - m.Qinj)

# Voltage magnitude constraints (PV buses): n_pv equations
pv_in_nr = [non_ref.index(i) for i in pv]
Vm_pv_sq = np.abs(V[pv]) ** 2
m.Vm_sq = Param("Vm_sq", Vm_pv_sq)
e_pv = m.e[pv_in_nr[0]:pv_in_nr[-1] + 1]
f_pv = m.f[pv_in_nr[0]:pv_in_nr[-1] + 1]
m.V_eqn = Eqn("V_pv", e_pv ** 2 + f_pv ** 2 - m.Vm_sq)

# %% Solve
spf, y0 = m.create_instance()
mdl = made_numerical(spf, y0, sparse=True)
sol = nr_method(mdl, y0)

# Reconstruct full voltage
e_sol = np.zeros(nb)
f_sol = np.zeros(nb)
e_sol[ref] = e_ref
f_sol[ref] = f_ref
e_sol[non_ref] = sol.y["e"]
f_sol[non_ref] = sol.y["f"]
V_sol = e_sol + 1j * f_sol

print(f"Converged in {sol.stats.nstep} iterations")
print(f"|V| = {np.abs(V_sol[non_ref])}")
print(f"Va  = {np.angle(V_sol[non_ref])}")
