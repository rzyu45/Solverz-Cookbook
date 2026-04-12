"""District Heating Network (DHS) hydraulic model — two formulations.

This module provides the *hydraulic subproblem* of a district heating
system (pipe mass flow distribution given node injection demand) written
in two equivalent styles using Solverz:

1. :func:`build_elementwise_model` — scalar ``Eqn`` per node / per loop,
   using indexed variables. This is the classical style used throughout
   the cookbook.

2. :func:`build_matmul_model` — compact matrix-vector formulation based
   on the node-pipe incidence matrix ``V`` and the loop incidence
   ``L``, derived from Yu et al., *Non-Iterative Calculation of
   Quasi-Dynamic Energy Flow in the Heat and Electricity Integrated
   Energy Systems*, IEEE Trans. Power Syst. 38(5), 2023.

Both formulations share the same governing equations:

    V m  =  m_inj         (mass continuity, one eqn per non-slack node)
    L · (K ⊙ m ⊙ |m|) = 0 (loop pressure, one eqn per loop)

where:

- ``V`` — node-pipe signed incidence matrix (n_node × n_pipe).
  ``V[i, j] = +1`` if pipe *j* flows into node *i*, ``-1`` if pipe *j*
  flows out of node *i*, ``0`` otherwise.
- ``m`` — pipe mass flow vector (n_pipe,), the unknown.
- ``m_inj`` — node injection vector: negative for source/slack nodes,
  positive for load nodes, zero for intermediate nodes.
- ``L`` — loop-pipe incidence (n_loop × n_pipe). Each row encodes a
  directed loop: ``L[ℓ, j] = ±1`` if pipe *j* is part of loop *ℓ*, the
  sign indicating the reference direction.
- ``K`` — per-pipe quadratic resistance coefficient.
- ``⊙`` — element-wise (Hadamard) product; ``|·|`` is element-wise abs.

The Mat_Mul formulation exercises Solverz's mutable-matrix Jacobian
code path (``diag(v) @ M`` and ``Diag(·)`` terms) end-to-end, making
this example a regression test for the matrix-vector calculus
infrastructure.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csc_array

from Solverz import Var, Eqn, Model, Param, Mat_Mul, Abs


def build_elementwise_model(hc: dict, m_init: np.ndarray, m_inj: np.ndarray):
    """Build the hydraulic subproblem using one scalar ``Eqn`` per node /
    per loop.

    Parameters
    ----------
    hc : dict
        DHS case dictionary (from ``SolUtil.sysparser.load_hs``) with
        fields ``n_node``, ``n_pipe``, ``G`` (networkx DiGraph), ``K``,
        ``pinloop``, ``slack_node``.
    m_init : ndarray (n_pipe,)
        Initial guess for the pipe mass flow vector.
    m_inj : ndarray (n_node,)
        Node injection vector (signed). ``V @ m`` should equal ``m_inj``.

    Returns
    -------
    Model
        A Solverz ``Model`` ready for ``create_instance``.
    """
    model = Model()
    model.m = Var('m', m_init)
    model.K = Param('K', hc['K'])
    model.m_inj = Param('m_inj', m_inj)

    slack_nodes = sorted(hc['slack_node'].tolist())
    n_node = hc['n_node']
    n_pipe = hc['n_pipe']

    # Mass continuity: we drop exactly ONE redundant row. For a
    # connected graph with all node injections prescribed the balance
    # system ``V m = m_inj`` has a single linearly dependent row
    # (summing every row gives ``0 = 0``), no matter how many slack
    # nodes are declared. Dropping one slack row is sufficient and
    # necessary; dropping one row per slack would make the system
    # underdetermined when there are multiple slacks.
    skip_node = slack_nodes[0] if slack_nodes else None
    for node in range(n_node):
        if node == skip_node:
            continue
        rhs = -model.m_inj[node]
        for edge in hc['G'].in_edges(node, data=True):
            pipe = edge[2]['idx']
            rhs = rhs + model.m[pipe]
        for edge in hc['G'].out_edges(node, data=True):
            pipe = edge[2]['idx']
            rhs = rhs - model.m[pipe]
        model.__dict__[f'mass_balance_{node}'] = Eqn(
            f'mass_balance_{node}', rhs)

    # Loop pressure: one scalar equation per independent loop.
    # ``hc['pinloop']`` can be a 1-D row vector (single loop) or a 2-D
    # array shaped ``(n_loop, n_pipe)``. Radial networks
    # (``n_loop == 0``) get no loop equation — the balance rows alone
    # are sufficient.
    pinloop = np.atleast_2d(np.asarray(hc['pinloop']))
    if pinloop.shape[1] != n_pipe and pinloop.shape[0] == n_pipe:
        pinloop = pinloop.T
    for loop_idx in range(pinloop.shape[0]):
        loop_row = pinloop[loop_idx]
        if not np.any(loop_row != 0):
            # Purely radial row — skip, no pressure constraint.
            continue
        rhs = 0
        for j in range(n_pipe):
            coeff = int(loop_row[j])
            if coeff != 0:
                rhs = rhs + coeff * model.K[j] * model.m[j] * Abs(model.m[j])
        model.__dict__[f'loop_pressure_{loop_idx}'] = Eqn(
            f'loop_pressure_{loop_idx}', rhs)

    return model


def build_matmul_model(hc: dict, m_init: np.ndarray, m_inj: np.ndarray):
    """Build the hydraulic subproblem using matrix-vector form.

    The two governing equations are a single vector equation each:

        V_ns @ m - m_inj_ns = 0   (one vector eqn of length n_node-1)
        L @ (K ⊙ m ⊙ |m|) = 0     (one vector eqn of length n_loop)

    where ``V_ns`` is ``V`` with the slack-node row removed and
    ``m_inj_ns`` the matching slice of the injection vector. ``L`` is
    the loop incidence built from ``hc['pinloop']``.

    The Mat_Mul form produces analytical Jacobian blocks of the shapes
    Solverz's matrix-calculus engine recognises:

    - ``∂(V_ns @ m) / ∂m = V_ns`` — constant-matrix derivative.
    - ``∂(L @ (K ⊙ m ⊙ |m|)) / ∂m = L · (diag(K ⊙ |m|) + diag(K ⊙ m ⊙
      sign(m)))`` — a *mutable matrix* Jacobian block (depends on m),
      triggering the ``Diag(v)@L`` scatter-add code path.
    """
    model = Model()
    model.m = Var('m', m_init)

    # --- Signed node-pipe incidence V (sparse), one slack row removed ---
    n_node = hc['n_node']
    n_pipe = hc['n_pipe']
    V_dense = np.zeros((n_node, n_pipe))
    for edge in hc['G'].edges(data=True):
        fnode, tnode, data = edge
        pipe = data['idx']
        V_dense[tnode, pipe] = 1      # flows INTO tnode
        V_dense[fnode, pipe] = -1     # flows OUT of fnode
    # Drop exactly one row — see :func:`build_elementwise_model` for the
    # rationale. Multi-slack systems drop only the first slack row.
    slack_nodes = sorted(hc['slack_node'].tolist())
    skip_node = slack_nodes[0] if slack_nodes else 0
    non_slack_rows = [n for n in range(n_node) if n != skip_node]
    V_ns = csc_array(V_dense[non_slack_rows, :])

    model.V_ns = Param('V_ns', V_ns, dim=2, sparse=True)
    model.m_inj_ns = Param('m_inj_ns', m_inj[non_slack_rows])
    model.K = Param('K', hc['K'])

    # --- Equation 1: mass continuity on non-slack nodes ---
    model.mass_balance = Eqn(
        'mass_balance',
        Mat_Mul(model.V_ns, model.m) - model.m_inj_ns)

    # --- Equation 2: loop pressure drop, one row per independent loop ---
    # Radial networks contribute no loop equations at all — the balance
    # system alone is square in that case.
    pinloop = np.atleast_2d(np.asarray(hc['pinloop']))
    if pinloop.shape[1] != n_pipe and pinloop.shape[0] == n_pipe:
        pinloop = pinloop.T
    # Strip any all-zero rows (radial sub-loops) before passing to the
    # model so the resulting L has exactly the loops we actually need.
    nontrivial_rows = [i for i in range(pinloop.shape[0])
                       if np.any(pinloop[i] != 0)]
    if nontrivial_rows:
        L_sparse = csc_array(pinloop[nontrivial_rows].astype(np.float64))
        model.L = Param('L', L_sparse, dim=2, sparse=True)
        model.loop_pressure = Eqn(
            'loop_pressure',
            Mat_Mul(model.L, model.K * model.m * Abs(model.m)))

    return model
