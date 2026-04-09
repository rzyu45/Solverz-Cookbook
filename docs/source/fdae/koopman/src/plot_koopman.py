import os

import matplotlib.pyplot as plt
import numpy as np

from Solverz import Eqn, Model, Opt, TimeSeriesParam, Var, fdae_solver, made_numerical
from Solverz.variable.ssymbol import AliasVar

current_dir = os.getcwd()
data_path = os.path.join(current_dir, "test_koopman", "koopman_data.npz")

BASEVALUE_P = 5e6
BASEVALUE_M = 10.0
N_TRAIN = 700
N_TEST = 300
N = N_TRAIN + N_TEST
DT = 1.0

# %% Load dataset
data = np.load(data_path)
pin_norm = data["pin"]
pout_norm = data["pout"]
qin_norm = data["qin"]
qout_norm = data["qout"]

# %% Build observables
x_data = np.column_stack(
    [
        pout_norm,
        qout_norm,
        (-pout_norm) * np.exp(-pout_norm),
        np.exp(-pout_norm) * np.sin(-pout_norm),
    ]
)
u_data = np.column_stack([pin_norm, qin_norm])

# %% Koopman regression
z_reg = np.hstack([x_data[: N_TRAIN - 1], u_data[1:N_TRAIN]])
k_all = (np.linalg.pinv(z_reg) @ x_data[1:N_TRAIN]).T
kx = k_all[:, :4]
ku = k_all[:, 4:]

# %% Solverz rollout
x0_test = x_data[N_TRAIN]
u_future = u_data[N_TRAIN + 1 :]
t_pred = len(u_future)
t_series = np.arange(t_pred + 1, dtype=float) * DT
u_p_series = np.concatenate([[u_future[0, 0]], u_future[:, 0]])
u_m_series = np.concatenate([[u_future[0, 1]], u_future[:, 1]])

model = Model()
model.x = Var("x", value=x0_test)
model.x_prev = AliasVar("x", step=1, value=x0_test)
model.u_P = TimeSeriesParam("u_P", v_series=u_p_series, time_series=t_series)
model.u_M = TimeSeriesParam("u_M", v_series=u_m_series, time_series=t_series)

for i in range(4):
    kx_row_sum = sum(kx[i, j] * model.x_prev[j] for j in range(4))
    ku_u = ku[i, 0] * model.u_P + ku[i, 1] * model.u_M
    model.__dict__[f"eq_{i}"] = Eqn(f"eq_{i}", model.x[i] - kx_row_sum - ku_u)

symbolic_model, y0 = model.create_instance()
numerical_model = made_numerical(symbolic_model, y0, sparse=True)
sol = fdae_solver(numerical_model, [0, t_pred * DT], y0, Opt(step_size=DT, pbar=False))
x_pred = sol.Y["x"]

# %% Plot
x_true = x_data[N_TRAIN : N_TRAIN + len(x_pred)]
rmse = np.sqrt(np.mean((x_true - x_pred) ** 2))

obs_labels = ["Pout linear", "Mout linear", "-Pout exp(-Pout)", "exp(-Pout) sin(-Pout)"]
t_all = np.arange(N)
t_pred_idx = np.arange(N_TRAIN, N_TRAIN + len(x_pred))

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
for idx, ax in enumerate(axes):
    ax.plot(t_all, x_data[:, idx], "k-", lw=1.5, label="True value")
    ax.plot(t_pred_idx, x_pred[:, idx], "r--", lw=1.5, label="Prediction")
    ax.axvline(N_TRAIN, color="g", ls="-.", lw=1.5, label="Train/test boundary")
    ax.set_ylabel(obs_labels[idx])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

axes[-1].set_xlabel("Time step k")
fig.suptitle(f"Koopman + Solverz prediction (RMSE = {rmse:.4e})", fontsize=13)
plt.tight_layout()
plt.show()
