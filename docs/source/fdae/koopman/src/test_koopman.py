import numpy as np

from plot_koopman import N_TRAIN, prepare_koopman_case, rollout_with_solverz


def test_koopman(datadir):
    x_data, _, kx, ku, x0_test, u_future = prepare_koopman_case()
    x_pred = rollout_with_solverz(kx, ku, x0_test, u_future)

    with open(datadir / "x_bench.npy", "rb") as f:
        x_bench = np.load(f)

    np.testing.assert_allclose(x_pred, x_bench, rtol=1e-6, atol=1e-8)

    x_true = x_data[N_TRAIN : N_TRAIN + len(x_pred)]
    assert np.sqrt(np.mean((x_true - x_pred) ** 2)) < 1e-2
