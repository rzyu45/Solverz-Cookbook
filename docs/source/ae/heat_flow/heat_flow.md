(heat_flow)=

# District Heating Network Hydraulic Flow

*Author: [Ruizhi Yu](https://github.com/rzyu45)*

This example solves the **hydraulic subproblem** of a district heating
system (DHS): given each node's mass injection demand, compute the
steady-state pipe mass flow distribution so that mass is conserved at
every node and the pressure drop around every loop is zero.

The example is written in two equivalent forms — one element-wise and
one using `Mat_Mul` — so it serves both as a worked walkthrough of the
matrix-vector calculus pipeline and as a regression test against
[SolUtil's `DhsFlow`](https://github.com/rzyu45/SolUtil) on the
well-known *Barry Island* 35-node network.

## Governing equations

Let $n_\text{node}$ be the number of nodes, $n_\text{pipe}$ the number
of pipes, and $n_\text{loop} = n_\text{pipe} - (n_\text{node} - 1)$ the
number of independent loops. For Barry Island, the counts are
$n_\text{node} = 35$, $n_\text{pipe} = 35$, $n_\text{loop} = 1$.

### Mass continuity

Let $V \in \mathbb{R}^{n_\text{node} \times n_\text{pipe}}$ be the
signed node–pipe incidence matrix:

```{math}
V_{ij} \;=\;
\begin{cases}
+1 & \text{if pipe } j \text{ flows into node } i, \\
-1 & \text{if pipe } j \text{ flows out of node } i, \\
0  & \text{otherwise.}
\end{cases}
```

Then mass balance at every node reads $V\,m = m^\text{inj}$, where
$m \in \mathbb{R}^{n_\text{pipe}}$ is the pipe-flow vector and
$m^\text{inj} \in \mathbb{R}^{n_\text{node}}$ is the signed node
injection (negative at sources/slacks, positive at loads, zero at
intermediate nodes).

For a connected network the rows of $V$ satisfy
$\mathbf{1}^\top V = 0$, so one equation is redundant. We drop the
slack row and keep $n_\text{node} - 1$ independent balance equations.

### Loop pressure

Let $L \in \mathbb{R}^{n_\text{loop} \times n_\text{pipe}}$ be the
loop–pipe incidence matrix: $L_{\ell j} = \pm 1$ if pipe $j$ belongs to
loop $\ell$, with sign giving the reference direction. For each
loop, the head loss around the closed contour is zero:

```{math}
L\,\bigl(K \odot m \odot |m|\bigr) \;=\; 0,
```

where $K \in \mathbb{R}^{n_\text{pipe}}$ is the per-pipe quadratic
resistance coefficient and $|\cdot|$ is element-wise absolute value.
The $m \odot |m| = m^2 \cdot \operatorname{sign}(m)$ form handles
signed flow directions correctly.

## Mat_Mul form

Putting the two together, the Mat_Mul model has **one vector equation
of length $n_\text{node} - 1$** and **one vector equation of length
$n_\text{loop}$**:

```{literalinclude} src/heat_flow_mdl.py
:pyobject: build_matmul_model
```

The symbolic Jacobian derived by the matrix-calculus engine is:

```{math}
\frac{\partial}{\partial m}\bigl(V_\text{ns}\,m - m^\text{inj}_\text{ns}\bigr)
\;=\; V_\text{ns},
```

```{math}
\frac{\partial}{\partial m}\bigl(L\,(K \odot m \odot |m|)\bigr)
\;=\; L \cdot \bigl(
    \operatorname{diag}(K \odot |m|)
    \;+\;
    \operatorname{diag}(K \odot m \odot \operatorname{sign}(m))
\bigr).
```

The first block is a **constant matrix Jacobian** (just $V_\text{ns}$).
The second is a **mutable matrix Jacobian** — it depends on $m$ — and
Solverz decomposes it into scatter-add loops at code-generation time.

## Element-wise form

For comparison, the same system written with one scalar equation per
node and per loop:

```{literalinclude} src/heat_flow_mdl.py
:pyobject: build_elementwise_model
```

Both formulations produce numerically identical pipe mass flows.

## Regression test

The test `test_heat_flow_hydraulic_matmul_regression` solves the
hydraulic system via all four code paths through Solverz
(inline/module × element-wise/Mat_Mul) and checks that each one
matches the ground truth from SolUtil's `DhsFlow`:

```{literalinclude} src/test_heat_flow.py
:pyobject: test_heat_flow_hydraulic_matmul_regression
```

The second test `test_heat_flow_stepwise_jacobian_match` is the
*strict* check that caught the v0.8.0 bug where the module printer
froze its mutable-matrix Jacobian blocks at their initial values. It
drives Newton iteration with the inline model and, at each step,
asserts that the module printer's $F$ and $J$ match the inline
values to machine precision. If the mutable-matrix block assembly
path ever regresses, this test fails immediately on step 0 or 1
instead of only degrading the final solution's accuracy.

## References

The Mat_Mul formulation of DHS hydraulic and thermal equations
follows:

> Yu, Gu, Lu, Yao, Zhang, Ding, Lu. *Non-Iterative Calculation of
> Quasi-Dynamic Energy Flow in the Heat and Electricity Integrated
> Energy Systems.* IEEE Transactions on Power Systems, 38(5),
> 4148–4162 (2023).
> [doi:10.1109/TPWRS.2022.3210167](https://doi.org/10.1109/TPWRS.2022.3210167)

equations (11)–(17). SolUtil's `DhsFlow` implements the equivalent
element-wise form used as ground truth.
