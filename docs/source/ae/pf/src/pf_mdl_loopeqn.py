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
from Solverz import Idx, LoopEqn, Sum


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

    # Int Params carrying the "free bus" index maps.
    m.pv_pq_idx = Param('pv_pq_idx', pv_pq_arr, dtype=int)
    m.pq_idx = Param('pq_idx', pq_arr, dtype=int)

    # Int Params + value Params for the Vm / Va pin LoopEqns below.
    m.ref_pv_idx = Param('ref_pv_idx', ref_pv_arr, dtype=int)
    m.Vm_pinned = Param('Vm_pinned', Vm_init[ref_pv_arr])
    m.ref_idx = Param('ref_idx', ref_arr, dtype=int)
    m.Va_pinned = Param('Va_pinned', Va_init[ref_arr])

    # Bounded indices — ``Sum`` and ``LoopEqn`` pick up the range
    # automatically from the ``.upper`` attribute so there's no need
    # to repeat the count in each call. We use two separate outer
    # indices so ``LoopEqn`` can infer distinct ``n_outer`` for the
    # P and Q equations (``P_eqn`` iterates pv+pq buses, ``Q_eqn``
    # iterates pq buses only).
    i_p = Idx('i_p', npvpq)
    i_q = Idx('i_q', npq)
    j = Idx('j', nb)

    # Split into separate Sums, one per sparse walker (Case A only —
    # ``_collect_sparse_walkers`` rejects multi-walker Sums).
    body_P = (
        m.Vm_full[m.pv_pq_idx[i_p]] * Sum(
            m.Vm_full[j] * m.Gbus[m.pv_pq_idx[i_p], j]
            * cos(m.Va_full[m.pv_pq_idx[i_p]] - m.Va_full[j]),
            j,
        )
        + m.Vm_full[m.pv_pq_idx[i_p]] * Sum(
            m.Vm_full[j] * m.Bbus[m.pv_pq_idx[i_p], j]
            * sin(m.Va_full[m.pv_pq_idx[i_p]] - m.Va_full[j]),
            j,
        )
        + m.Pd[m.pv_pq_idx[i_p]] - m.Pg[m.pv_pq_idx[i_p]]
    )
    m.P_eqn = LoopEqn('P_eqn', outer_index=i_p, body=body_P, model=m)

    body_Q = (
        m.Vm_full[m.pq_idx[i_q]] * Sum(
            m.Vm_full[j] * m.Gbus[m.pq_idx[i_q], j]
            * sin(m.Va_full[m.pq_idx[i_q]] - m.Va_full[j]),
            j,
        )
        - m.Vm_full[m.pq_idx[i_q]] * Sum(
            m.Vm_full[j] * m.Bbus[m.pq_idx[i_q], j]
            * cos(m.Va_full[m.pq_idx[i_q]] - m.Va_full[j]),
            j,
        )
        + m.Qd[m.pq_idx[i_q]] - m.Qg[m.pq_idx[i_q]]
    )
    m.Q_eqn = LoopEqn('Q_eqn', outer_index=i_q, body=body_Q, model=m)

    # Pin ref+pv buses' Vm and ref buses' Va to their known values
    # via two tiny LoopEqns instead of per-index scalar ``Eqn``s.
    # The ``Vm_pin`` body uses an indirect-outer indexing pattern
    # (``Vm_full[ref_pv_idx[i_vp]]``) which the Phase J indirect
    # walker handles: F = Vm_full[ref_pv_idx[i_vp]] - Vm_pinned[i_vp];
    # derivative wrt Vm_full is a ``δ(ref_pv_idx[i_vp], k)`` sparsity
    # pattern that the J-side analyzer recognises as indirect diag.
    nref_pv = len(ref_pv_arr)
    nref = len(ref_arr)
    i_vp = Idx('i_vp', nref_pv)
    i_vr = Idx('i_vr', nref)

    m.Vm_pin = LoopEqn(
        'Vm_pin', outer_index=i_vp,
        body=m.Vm_full[m.ref_pv_idx[i_vp]] - m.Vm_pinned[i_vp],
        model=m,
    )
    m.Va_pin = LoopEqn(
        'Va_pin', outer_index=i_vr,
        body=m.Va_full[m.ref_idx[i_vr]] - m.Va_pinned[i_vr],
        model=m,
    )

    return m


if __name__ == '__main__':
    m = build_loopeqn_pf_model()
    spf, y0 = m.create_instance()
    pyprinter = module_printer(spf, y0, 'powerflow_loopeqn',
                               jit=True)
    pyprinter.render()
