import numpy as np
from Solverz import Var, Param, Eqn, made_numerical, sin, Model, nr_method, Opt

import matplotlib.pyplot as plt

# %% modelling
m = Model()
m.x = Var('x', 0)
m.t = Param('t', 0.3)
m.x1 = Param('x1', -0.1)
m.f = Eqn('f', m.x - sin(np.pi * m.x) * m.t - m.x1)
sae, y0 = m.create_instance()
ae, code = made_numerical(sae, y0, output_code=True, sparse=True)
# %% solution
opt = Opt(ite_tol=1e-8)
X = np.linspace(-1, 1, 81)
U = np.zeros((81, 5))
t_range = [0.1, 0.3, 0.5, 0.7, 1]
tshock = 0.31831
for j in range(5):
    ae.p['t'] = t_range[j]
    for i in range(X.shape[0]):
        ae.p['x1'] = X[i]
        if t_range[j] < tshock:
            sol = nr_method(ae, y0, opt)
            U[i, j] = -np.sin(np.pi * sol.y['x'])[0]
        else:
            if X[i] > 0:
                y0['x'] = 1
                sol = nr_method(ae, y0, opt)
                U[i, j] = -np.sin(np.pi * sol.y['x'])[0]
            elif X[i] < 0:
                y0['x'] = -1
                sol = nr_method(ae, y0, opt)
                U[i, j] = -np.sin(np.pi * sol.y['x'])[0]
            else:
                U[i, j] = 0

# %% visualize
plt.plot(X, U, label=[f"t={arg}" for arg in t_range])
plt.xlabel('x')
plt.ylabel('u')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.legend()
plt.grid()
plt.show()
