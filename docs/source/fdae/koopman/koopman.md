(koopman)=

# Solving Gas Transmission PDEs using Koopman Operator Theory

*Author: [Yuan Li](https://github.com/yuanyuan11234)*

The natural gas pipeline model is a nonlinear dynamic system driven by the boundary pressure and mass flow rate. Instead of solving the governing partial differential equations (PDEs) directly, we build a data-driven surrogate with Koopman operator theory and then use Solverz to perform multi-step prediction.

At each discrete time step $k$, four signals are measured:

| Signal | Role | Base value |
| --- | --- | --- |
| $P_\text{in}(k)$ | inlet pressure input | $5\times10^6$ Pa |
| $P_\text{out}(k)$ | outlet pressure output | $5\times10^6$ Pa |
| $m_\text{in}(k)$ | inlet mass flow input | $10$ kg/s |
| $m_\text{out}(k)$ | outlet mass flow output | $10$ kg/s |

All signals are normalized by their base values. The dataset contains $N=1000$ samples, with the first $N_\text{train}=700$ samples used for regression and the remaining ones used for prediction.

## Observable Space

We define the lifted observable vector as

```{math}
:label: obs
\boldsymbol{\psi}(k)=
\begin{bmatrix}
P_\text{out}(k)\\
m_\text{out}(k)\\
\qty(-P_\text{out}(k))e^{-P_\text{out}(k)}\\
e^{-P_\text{out}(k)}\sin\qty(-P_\text{out}(k))
\end{bmatrix}\in\mathbb{R}^4.
```

The first two entries are linear observables and the last two entries are nonlinear lifting terms used to capture the gas-flow dynamics.

## Koopman Regression

Let $\mathbf{X}\in\mathbb{R}^{N\times4}$ collect the observables and $\mathbf{U}\in\mathbb{R}^{N\times2}$ collect the normalized inputs. The Koopman model is written as

```{math}
:label: koopman
\boldsymbol{\psi}(k)=K_x\boldsymbol{\psi}(k-1)+K_u\mathbf{u}(k),
```

where $K_x\in\mathbb{R}^{4\times4}$ and $K_u\in\mathbb{R}^{4\times2}$ are constant matrices identified from training data.

Define the regression matrix over the training window as

```{math}
:label: regression
Z=\qty[\mathbf{X}_{0:T-1},\;\mathbf{U}_{1:T}]\in\mathbb{R}^{(T-1)\times6},
```

then the least-squares solution is

```{math}
\begin{bmatrix}K_x\mid K_u\end{bmatrix}^{\mathsf{T}}=Z^\dagger \mathbf{X}_{1:T}.
```

## Solverz Formulation

To run multi-step prediction in Solverz, we rewrite {eq}`koopman` into a residual equation for each state component:

```{math}
:label: solverz_res
F_i\bigl(\mathbf{x}(k),\mathbf{x}(k-1),\mathbf{u}(k)\bigr)
=x_i(k)-\sum_{j=0}^{3}K_x[i,j]x_j(k-1)-K_u[i,0]u_P(k)-K_u[i,1]u_M(k)=0,
```

for $i=0,\ldots,3$, where $\mathbf{x}(k)\equiv\boldsymbol{\psi}(k)$.

The delayed state $\mathbf{x}(k-1)$ is represented by `AliasVar`, so Solverz can automatically substitute the state value from the previous time step.

The complete implementation is shown below. The source file and the lightweight benchmark data package are stored in the [same directory on GitHub](https://github.com/rzyu45/Solverz-Cookbook/tree/main/docs/source/fdae/koopman).

```{literalinclude} src/plot_koopman.py
:language: python
```

The prediction result is

```{eval-rst}
.. plot:: fdae/koopman/src/plot_koopman.py
```

The learned Koopman model tracks the test trajectory well, and the plotted curves show that the Solverz rollout reproduces the main trends of all lifted observables.
