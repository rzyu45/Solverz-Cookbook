import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from powerflow import mdl as pf, y as y0
from Solverz import nr_method, sicnm, Opt

current_dir = os.getcwd()
file_path = os.path.join(current_dir, "test_pf_jac", "ill-Va.mat")
mat = loadmat(file_path)

pv = (mat['pv'] - 1).reshape(-1).tolist()
pq = (mat['pq'] - 1).reshape(-1).tolist()
Va0 = np.deg2rad(mat['Va0'])

for i in range(len(pv + pq)):
    y0['Va'][i:i + 1] = Va0[(pv + pq)[i]]

# Failed, cannot converge within 100 iterations
sol = nr_method(pf, y0, Opt(ite_tol=1e-5))
# Succeeded
sol1 = sicnm(pf, y0, Opt(scheme='rodas3d', rtol=1e-1, atol=1e-1, hinit=0.1, ite_tol=1e-5))


Vm_bench = np.array([0.98313825, 0.98009295, 0.98240617, 0.97318397, 0.9673554,
                     0.96062366, 0.98050609, 0.98440428, 0.98050609, 0.9854683,
                     0.97667682, 0.98022901, 0.97739562, 0.97686538, 0.96844031,
                     0.96528702, 0.96916633, 0.99338328, 0.9885663, 0.99021484,
                     0.97219415, 0.97471485, 0.9795967, 0.96788288])
# %% visualize error
plt.title(r'$\Delta |V_\text{m}|$')
plt.plot(np.abs(sol1.y['Vm']-Vm_bench))
plt.show()
