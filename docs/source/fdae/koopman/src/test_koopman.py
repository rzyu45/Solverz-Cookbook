import numpy as np

from Solverz import Eqn, Model, Opt, TimeSeriesParam, Var, fdae_solver, made_numerical
from Solverz.variable.ssymbol import AliasVar

N_TRAIN = 700
DT = 1.0


def test_koopman(datadir):
    data = np.load(datadir / "koopman_data.npz")
    pin_norm = data["pin"]
    pout_norm = data["pout"]
    qin_norm = data["qin"]
    qout_norm = data["qout"]

    x_data = np.column_stack(
        [
            pout_norm,
            qout_norm,
            (-pout_norm) * np.exp(-pout_norm),
            np.exp(-pout_norm) * np.sin(-pout_norm),
        ]
    )
    u_data = np.column_stack([pin_norm, qin_norm])

    z_reg = np.hstack([x_data[: N_TRAIN - 1], u_data[1:N_TRAIN]])
    k_all = (np.linalg.pinv(z_reg) @ x_data[1:N_TRAIN]).T
    kx = k_all[:, :4]
    ku = k_all[:, 4:]

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

    with open(datadir / "x_bench.npy", "rb") as f:
        x_bench = np.load(f)

    np.testing.assert_allclose(x_pred, x_bench, rtol=1e-4, atol=1e-5)

    x_true = x_data[N_TRAIN : N_TRAIN + len(x_pred)]
    assert np.sqrt(np.mean((x_true - x_pred) ** 2)) < 1e-2
