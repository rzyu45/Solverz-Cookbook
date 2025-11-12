import numpy as np
from Solverz import Var, Eqn, Ode, Param, Model, module_printer, exp, ln, sin, cos, Saturation, TimeSeriesParam, Rodas, \
    Opt, made_numerical
import scipy.io as sio
from sympy import re as real, im as imag
import pytest
import sys
import importlib
from pathlib import Path


@pytest.fixture
def dynamic_module(request, datadir, tmp_path):
    """动态生成求解器模块并自动清理的fixture"""
    # ========== 模块配置 ==========
    module_name = "ieee33pv"
    module_dir = tmp_path / "Solverz_dynamic_modules"
    module_dir.mkdir()

    # ========== 模型初始化 ==========
    par = sio.loadmat(datadir / 'par.mat')
    y = sio.loadmat(datadir / 'y0.mat')['y'].reshape((-1,))

    m = Model()
    m.delta = Var('delta', y[0:2])
    m.omega = Var('omega', y[2:4])
    m.ix_pv = Var('ix_pv', -y[4:8])
    m.iy_pv = Var('iy_pv', -y[8:12])
    m.idref1 = Var('idref1', y[12:16])
    m.urd1 = Var('urd1', y[16:20])
    m.urq1 = Var('urq1', y[20:24])
    m.upv = Var('upv', y[24:28])
    m.iL = Var('iL', y[28:32])
    m.udc = Var('udc', y[32:36])
    m.D1 = Var('D1', y[36:40])
    m.Ux = Var('Ux', y[40:73])
    m.Uy = Var('Uy', y[73:106])
    m.Ix = Var('Ix', y[106:139])
    m.Iy = Var('Iy', y[139:172])

    for name in par.keys():
        if name not in ['G', 'B']:
            if isinstance(par[name], np.ndarray):
                m.__dict__[name] = Param(name, par[name].reshape(-1))
    m.IB_sys = Param('IB_sys', 2.28)
    m.Radiation = Param('Radiation', [1000, 1000, 1000, 1000])
    m.kp = Param('kp', -m.kp.value)
    m.ki = Param('ki', -m.ki.value)
    m.kop = Param('kop', -m.kop.value)
    m.koi = Param('koi', -m.koi.value)
    m.Sfluc = TimeSeriesParam('Sfluc',
                              [1, 1, 1, 1, 1, 1],
                              [0, 4.99999, 5, 6, 6.00001, 20])

    G = par['G'].toarray()
    B = par['B'].toarray()

    isc = m.ISC * m.Radiation * m.Sfluc / m.sref * (1 + 0.0025 * (m.T - 25))
    im = m.IM * m.Radiation * m.Sfluc / m.sref * (1 + 0.0025 * (m.T - 25))
    uoc = m.UOC * (1 - 0.00288 * (m.T - 25)) * ln(exp(1) + 0.5 * (m.Radiation * m.Sfluc / 1000 - 1))
    um = m.UM * (1 - 0.00288 * (m.T - 25)) * ln(exp(1) + 0.5 * (m.Radiation * m.Sfluc / 1000 - 1))
    c2 = (um / uoc - 1) / ln(1 - im / isc)
    c1 = (1 - im / isc) * exp(-um / (c2 * uoc))

    idx_pv = [7, 12, 27, 31]
    m.ux_pv = Var('ux_pv', m.Ux.value[idx_pv])
    m.uy_pv = Var('uy_pv', m.Uy.value[idx_pv])
    for i in range(4):
        m.__dict__[f'eqn_usd_{i}'] = Eqn(f'eqn_usd_{i}', m.ux_pv[i] - m.Ux[idx_pv[i]])
        m.__dict__[f'eqn_usq_{i}'] = Eqn(f'eqn_usq_{i}', m.uy_pv[i] - m.Uy[idx_pv[i]])
        m.__dict__[f'eqn_ixpv_{i}'] = Eqn(f'eqn_ixpv_{i}',
                                          m.Ix[idx_pv[i]] - m.ix_pv[i] * m.Inom / m.IB_sys)
        m.__dict__[f'eqn_iypv_{i}'] = Eqn(f'eqn_iypv_{i}',
                                          m.Iy[idx_pv[i]] - m.iy_pv[i] * m.Inom / m.IB_sys)

    ugrid_pv = m.ux_pv + 1j * m.uy_pv
    exp_j_theta = ugrid_pv / abs(ugrid_pv)
    usk_ctrl = ugrid_pv / exp_j_theta
    usdk = real(usk_ctrl)
    usqk = imag(usk_ctrl)

    ig = m.ix_pv + 1j * m.iy_pv
    ig_ctrl = ig / exp_j_theta
    id = real(ig_ctrl)
    iq = imag(ig_ctrl)

    idref = Saturation(m.kop * (m.udcref - m.udc) + m.koi * m.idref1, -1, 1)
    iqref = 0

    urd = usdk - m.ws * m.lf * iq + m.kip * (idref - id) + m.kii * m.urd1
    urq = usqk + m.ws * m.lf * id + m.kip * (iqref - iq) + m.kii * m.urq1
    urdq = urd + 1j * urq
    K_temp = 380 * 2 * np.sqrt(2 / 3)
    temp = Saturation(K_temp / m.udc, 0, 1)
    uidq = (1 / K_temp) * temp * m.udc * urdq
    uid = real(uidq)
    uiq = imag(uidq)
    uixy = uidq * exp_j_theta
    uix = real(uixy)
    uiy = imag(uixy)
    idc = (3 / 2) * m.Pnom * (uid * id + uiq * iq) / m.udc
    ipv = isc * (1 - c1 * (exp(m.upv / (c2 * uoc)) - 1))
    D = Saturation(m.kp * (um - m.upv) + m.ki * m.D1, 1e-6, 1 - 1e-6)

    m.eqn_delta = Ode('eqn_delta', m.ws * (m.omega - 1), m.delta)
    idx_gen = [5, 29]
    for i in range(2):
        igen = idx_gen[i]
        Pei = m.Ux[igen] * m.Ix[igen] + m.Uy[igen] * m.Iy[igen] + m.ra[i] * (m.Ix[igen] ** 2 + m.Iy[igen] ** 2)
        m.__dict__[f'eqn_omega_{i}'] = Ode(f'eqn_omega_{i}',
                                           (m.pm[i] - Pei - m.Damping[i] * (m.omega[i] - 1)) / m.Tj[i],
                                           m.omega[i])
        m.__dict__[f'eqn_Edp_{i}'] = Eqn(f'eqn_Edp_{i}',
                                         (m.Edp[i] - sin(m.delta[i]) * (
                                                 m.Ux[igen] + m.ra[i] * m.Ix[igen] - m.xqp[i] * m.Iy[igen])
                                          + cos(m.delta[i]) * (
                                                  m.Uy[igen] + m.ra[i] * m.Iy[igen] + m.xqp[i] * m.Ix[igen])))
        m.__dict__[f'eqn_Eqp_{i}'] = Eqn(f'eqn_Eqp_{i}',
                                         (m.Eqp[i] - cos(m.delta[i]) * (
                                                 m.Ux[igen] + m.ra[i] * m.Ix[igen] - m.xdp[i] * m.Iy[igen])
                                          - sin(m.delta[i]) * (
                                                  m.Uy[igen] + m.ra[i] * m.Iy[igen] + m.xdp[i] * m.Ix[igen])))

    m.eqn_id = Ode('eqn_id', m.ws / m.lf * (uix - m.ux_pv + m.lf * m.iy_pv), m.ix_pv)
    m.eqn_iq = Ode('eqn_iq', m.ws / m.lf * (uiy - m.uy_pv - m.lf * m.ix_pv), m.iy_pv)
    m.eqn_idref1 = Ode('eqn_idref1', m.udcref - m.udc, m.idref1)
    m.eqn_urd1 = Ode('eqn_urd1',
                     idref - id,
                     m.urd1)
    m.eqn_urq1 = Ode('eqn_urq1',
                     iqref - iq,
                     m.urq1)
    m.eqn_upv = Ode('eqn_upv',
                    1 / m.cpv * (ipv - m.iL),
                    m.upv)
    m.eqn_iL = Ode('eqn_iL',
                   1 / m.ldc * (m.upv - (1 - D) * m.udc),
                   m.iL)
    m.eqn_udc = Ode('eqn_udc',
                    1 / m.cdc * ((1 - D) * m.iL - idc),
                    m.udc)
    m.eqn_D1 = Ode('eqn_D1',
                   um - m.upv,
                   m.D1)

    nb = m.Ix.value.shape[0]
    for i in range(nb):
        rhs1 = m.Ix[i]
        rhs2 = m.Iy[i]
        for j in range(nb):
            rhs1 = rhs1 - (G[i, j] * m.Ux[j] - B[i, j] * m.Uy[j])
            rhs2 = rhs2 - (G[i, j] * m.Uy[j] + B[i, j] * m.Ux[j])
        m.__dict__[f'eqn_Ux_{i}'] = Eqn(f'eqn_Ux_{i}', rhs1)
        m.__dict__[f'eqn_Uy_{i}'] = Eqn(f'eqn_Uy_{i}', rhs2)

    m.Ux_slack = Eqn('eqn_Ux_slack', m.Ux[0] - 1)
    m.Uy_slack = Eqn('eqn_Uy_slack', m.Uy[0])

    for i in (par['id_others'].reshape(-1) - 1).tolist():
        m.__dict__[f'eqn_ix_{i}'] = Eqn(f'eqn_ix_{i}', m.Ix[i])
        m.__dict__[f'eqn_iy_{i}'] = Eqn(f'eqn_iy_{i}', m.Iy[i])

    # ========== 生成求解器模块 ==========
    sdae, y0 = m.create_instance()

    printer = module_printer(
        sdae,
        y0,
        module_name,
        directory=str(module_dir),
        jit=True
    )
    printer.render()

    # ========== 动态导入机制 ==========
    sys.path.insert(0, str(module_dir))
    request.addfinalizer(lambda: sys.path.remove(str(module_dir)))

    # 强制重新加载模块
    if module_name in sys.modules:
        mod = importlib.reload(sys.modules[module_name])
    else:
        mod = importlib.import_module(module_name)

    # 参数后处理
    mod.mdl.p['Sfluc'] = TimeSeriesParam(
        'Sfluc',
        [1, 1, 0.8, 0.8, 1, 1],
        [0, 4.99999, 5, 6, 6.00001, 20]
    )

    # ========== 自动清理保障 ==========
    def cleanup():
        if module_name in sys.modules:
            del sys.modules[module_name]
        # 临时目录由 pytest 自动清理

    request.addfinalizer(cleanup)

    return mod.mdl, mod.y


def test_ieee33pv(dynamic_module, datadir):
    """优化后的测试用例"""
    dae, y0 = dynamic_module

    # 执行求解
    sol = Rodas(
        dae,
        np.linspace(0, 20, 1001),
        y0,
        Opt(pbar=True)
    )

    # 结果验证
    U = (sol.Y['Ux'] ** 2 + sol.Y['Uy'] ** 2) ** 0.5
    P = sol.Y['Ux'] * sol.Y['Ix'] + sol.Y['Uy'] * sol.Y['Iy']

    # 加载基准数据
    with open(datadir / 'U_bench.npy', 'rb') as f:
        U_bench = np.load(f)
    with open(datadir / 'P_bench.npy', 'rb') as f:
        P_bench = np.load(f)

    # 断言验证
    np.testing.assert_allclose(U,
                               U_bench,
                               rtol=1e-4,
                               atol=1e-5)
    # dP = np.mean(np.abs(P[:, [5, 7, 12, 27, 29, 31]]-P_bench[:, [5, 7, 12, 27, 29, 31]]))
    # assert dP < 1e-6
    np.testing.assert_allclose(P[:, [5, 7, 12, 27, 29, 31]],
                               P_bench[:, [5, 7, 12, 27, 29, 31]],
                               rtol=1e-2,
                               atol=1e-3)
