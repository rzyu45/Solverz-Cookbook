(pf)=

# The Jacobian Calculation of Electric Power Flow

The Jacobian matrix is a fundamental quantity in AC power flow computation and sensitivity analysis of electric power systems.

In this example, we illustrate how to use Solverz to symbolically model the power flow and then derive the numerical Jacobian function.

## Model

The power flow models are equations describing the injection power of electric buses, with formulae

```{math}
\left\{
\begin{aligned}
&p_h=v_h\sum_{k}v_k\qty(g_{hk}\cos\theta_{hk}+b_{hk}\sin\theta_{hk}),\quad i\in\mathbb{B}_\text{pv, pq}\\
&q_h=v_h\sum_{k}v_k\qty(g_{hk}\sin\theta_{hk}-b_{hk}\cos\theta_{hk}),\quad i\in\mathbb{B}_\text{pv}\\
\end{aligned}
\right.
```

where $p$ and $q$ are respectively the active and reactive injection power; $v_h$ is the voltage magnitude of bus $h$; $\theta_{hk}=\theta_h-\theta_k$ is the voltage angle difference between bus $h$ and $k$; $g_{hk}$ and $b_{hk}$ are the $(h, k)$-th entry of the conductance and susceptance matrices; $\mathbb{B}$ is the set of bus indices with the subscripts being the bus types.

## Implementation in Solverz

We use the `case30` from the [matpower](https://matpower.org/) library. The required data for verification, the `pf.mat`, can be found in case file directory of the [source repo](https://github.com/rzyu45/Solverz-Cookbook).

The codes are

```{literalinclude} src/plot_pf.py
```

The sparse pattern of the Jacobian is visualized as follows.

```{eval-rst}
.. plot:: ae/pf/src/plot_pf.py
```
