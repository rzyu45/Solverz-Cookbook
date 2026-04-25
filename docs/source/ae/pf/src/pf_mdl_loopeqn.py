"""Polar power flow modelled with LoopEqn instead of the per-bus
scalar expansion used in ``pf_mdl.py``.

The LoopEqn variant keeps a flat ``Vm_full`` / ``Va_full`` over every
bus and pins the ref / pv portions with two *additional* LoopEqn
blocks (``Vm_pin`` over ref+pv, ``Va_pin`` over ref). Every
sub-function in the generated module is therefore either a LoopEqn
kernel or the F_/J_ wrapper — there are no per-scalar
``inner_F<N>``'s emitted for the pin equations.

This module is used by ``bench_pf_loopeqn_vs_legacy.py`` to measure the
cold-cache module_printer compile-time delta.
"""
import os

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_array

from Solverz import Var, Model, sin, cos, Param, module_printer
from Solverz import Idx, LoopEqn, Sum, Set


def build_loopeqn_pf_model(datadir=None):
    """Build the LoopEqn polar power flow model for case30.

    Parameters
    ----------
    datadir : str or None
        Directory containing ``pf.mat`` and ``pq.mat``. Defaults to the
        ``test_pf_jac`` folder next to this file.

    Returns
    -------
    m : Model
        Fully assembled Solverz ``Model`` with flat ``Vm_full`` /
        ``Va_full`` Vars, two ``LoopEqn``s (``P_eqn`` / ``Q_eqn``), and
        scalar pin Eqns for the ref / pv buses.
    """
    if datadir is None:
        datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'test_pf_jac')
    param = loadmat(os.path.join(datadir, 'pf.mat'))
    PQ = loadmat(os.path.join(datadir, 'pq.mat'))

    V = param['V'].reshape(-1)
    nb = V.shape[0]
    Ybus = param['Ybus']
    ref = (param['ref'] - 1).reshape(-1).tolist()
    pv = (param['pv'] - 1).reshape(-1).tolist()
    pq = (param['pq'] - 1).reshape(-1).tolist()

    mbase = 100  # MVA
    Pg = PQ['Pg'].reshape(-1) / mbase
    Qg = PQ['Qg'].reshape(-1) / mbase
    Pd = PQ['Pd'].reshape(-1) / mbase
    Qd = PQ['Qd'].reshape(-1) / mbase

    Vm_init = np.abs(V)
    Va_init = np.angle(V)

    pv_pq_arr = np.array(pv + pq, dtype=int)  # free buses for P balance
    pq_arr = np.array(pq, dtype=int)          # free buses for Q balance
    ref_pv_arr = np.array(ref + pv, dtype=int)
    ref_arr = np.array(ref, dtype=int)

    npvpq = len(pv_pq_arr)
    npq = len(pq_arr)

    m = Model()
    m.Vm_full = Var('Vm_full', Vm_init)
    m.Va_full = Var('Va_full', Va_init)

    # Sparse admittance matrices. The indirect-outer CSR walker
    # ``Gbus[pv_pq_idx[i], j]`` reads row ``pv_pq_idx[i]`` from the
    # CSR skeleton at each outer iteration — identical effort to the
    # direct-outer walker but over the subset of buses the LoopEqn
    # iterates. Both F-side emission and J-side sparsity analysis
    # honour the row map.
    m.Gbus = Param('Gbus', csc_array(Ybus.real), dim=2, sparse=True)
    m.Bbus = Param('Bbus', csc_array(Ybus.imag), dim=2, sparse=True)

    m.Pg = Param('Pg', Pg)
    m.Qg = Param('Qg', Qg)
    m.Pd = Param('Pd', Pd)
    m.Qd = Param('Qd', Qd)

    # Four index sets: the full bus set (``j`` sums), the free-bus
    # subsets (``pv_pq`` for P balance, ``pq`` for Q balance), and
    # the pinned-bus subsets (``ref_pv`` for Vm pins, ``ref`` for Va
    # pins). ``Set`` replaces the ``Param + Idx + m.map[i]`` dance
    # with a single named object — ``Set.idx(...)`` produces a
    # bounded sympy.Idx whose uses in the body are transparently
    # gathered via the set's auxiliary Param.
    m.Bus = Set('Bus', nb)
    m.PVPQ = Set('PVPQ', pv_pq_arr)
    m.PQ = Set('PQ', pq_arr)
    m.RefPV = Set('RefPV', ref_pv_arr)
    m.Ref = Set('Ref', ref_arr)

    m.Vm_pinned = Param('Vm_pinned', Vm_init[ref_pv_arr])
    m.Va_pinned = Param('Va_pinned', Va_init[ref_arr])

    i_p = m.PVPQ.idx('i_p')
    i_q = m.PQ.idx('i_q')
    j = m.Bus.idx('j')

    # Split into separate Sums, one per sparse walker (Case A only —
    # ``_collect_sparse_walkers`` rejects multi-walker Sums).
    body_P = (
        m.Vm_full[i_p] * Sum(
            m.Vm_full[j] * m.Gbus[i_p, j]
            * cos(m.Va_full[i_p] - m.Va_full[j]),
            j,
        )
        + m.Vm_full[i_p] * Sum(
            m.Vm_full[j] * m.Bbus[i_p, j]
            * sin(m.Va_full[i_p] - m.Va_full[j]),
            j,
        )
        + m.Pd[i_p] - m.Pg[i_p]
    )
    m.P_eqn = LoopEqn('P_eqn', outer_index=i_p, body=body_P, model=m)

    body_Q = (
        m.Vm_full[i_q] * Sum(
            m.Vm_full[j] * m.Gbus[i_q, j]
            * sin(m.Va_full[i_q] - m.Va_full[j]),
            j,
        )
        - m.Vm_full[i_q] * Sum(
            m.Vm_full[j] * m.Bbus[i_q, j]
            * cos(m.Va_full[i_q] - m.Va_full[j]),
            j,
        )
        + m.Qd[i_q] - m.Qg[i_q]
    )
    m.Q_eqn = LoopEqn('Q_eqn', outer_index=i_q, body=body_Q, model=m)

    # Pin ref+pv buses' Vm and ref buses' Va to their known values
    # via two tiny LoopEqns over the ``RefPV`` / ``Ref`` sets. The
    # body uses the set-tagged ``Idx`` directly; the rewriter
    # inserts the gather to the underlying bus index automatically.
    i_vp = m.RefPV.idx('i_vp')
    i_vr = m.Ref.idx('i_vr')

    m.Vm_pin = LoopEqn(
        'Vm_pin', outer_index=i_vp,
        body=m.Vm_full[i_vp] - m.Vm_pinned[i_vp],
        model=m,
    )
    m.Va_pin = LoopEqn(
        'Va_pin', outer_index=i_vr,
        body=m.Va_full[i_vr] - m.Va_pinned[i_vr],
        model=m,
    )

    return m


if __name__ == '__main__':
    m = build_loopeqn_pf_model()
    spf, y0 = m.create_instance()
    pyprinter = module_printer(spf, y0, 'powerflow_loopeqn',
                               jit=True)
    pyprinter.render()
