import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_plot_module():
    module_path = Path(__file__).with_name("plot_ies.py")
    spec = importlib.util.spec_from_file_location("plot_ies", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ies(datadir):
    module = load_plot_module()
    benchmark = pd.read_excel(datadir / "benchmark.xlsx")

    time_bench = benchmark["time"].to_numpy(dtype=float)
    qfuel_bench = benchmark["qfuel"].to_numpy(dtype=float)

    time, qfuel = module.get_qfuel_response(
        power_case=datadir / "caseI.xlsx",
        heat_case=datadir / "case_heat.xlsx",
    )

    assert time.shape == time_bench.shape
    assert qfuel.shape == qfuel_bench.shape

    np.testing.assert_allclose(time, time_bench, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(qfuel, qfuel_bench, rtol=1e-6, atol=1e-8)
