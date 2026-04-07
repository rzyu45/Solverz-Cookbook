import importlib.util
from pathlib import Path

import numpy as np


def load_plot_module():
    module_path = Path(__file__).with_name("plot_ies.py")
    spec = importlib.util.spec_from_file_location("plot_ies", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ies(datadir):
    module = load_plot_module()
    module.POWER_CASE = datadir / "caseI.xlsx"
    module.HEAT_CASE = datadir / "case_heat.xlsx"

    time, omega_gt, omega_st, omega_coi = module.get_frequency_response()

    assert time.ndim == 1
    assert omega_gt.shape == time.shape
    assert omega_st.shape == time.shape
    assert omega_coi.shape == time.shape

    assert np.isfinite(time).all()
    assert np.isfinite(omega_gt).all()
    assert np.isfinite(omega_st).all()
    assert np.isfinite(omega_coi).all()

    np.testing.assert_allclose(time[0], 0.0)
    np.testing.assert_allclose(time[-1], 10 * 3600)
    assert np.all(np.diff(time) > 0)

    np.testing.assert_allclose(omega_gt[0], 1.0, atol=5e-3)
    np.testing.assert_allclose(omega_st[0], 1.0, atol=5e-3)
    np.testing.assert_allclose(omega_coi[0], 1.0, atol=5e-3)

    assert np.max(np.abs(omega_gt - omega_gt[0])) > 1e-4
    assert np.max(np.abs(omega_st - omega_st[0])) > 1e-4
    assert np.max(np.abs(omega_coi - omega_coi[0])) > 1e-4

    final_min = min(omega_gt[-1], omega_st[-1])
    final_max = max(omega_gt[-1], omega_st[-1])
    assert final_min <= omega_coi[-1] <= final_max
