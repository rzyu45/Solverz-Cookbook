(pf)=

# The Calculation of Electric Power Flow

Power flow is fundamental in electric power system analysis. Given electric load and generation, we want to calculate 
the bus voltage and the power distribution[^book1].

In this example, we illustrate how to use Solverz to symbolically model the power flow and then perform the power flow
analysis.

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

The buses in electric power systems are typically sparsely connected, and hence the Jacobian of power flow models are always sparse. In what follows, we will set the `sparse` flag to be `True`.

## Implementation in Solverz

### Power flow modelling

We use the `case30` from the [matpower](https://matpower.org/) library. The required data for verification can be found in case file directory of the [source repo](https://github.com/rzyu45/Solverz-Cookbook).

We first perform the symbolic modelling of the `case30` power flow. 

```{eval-rst}
.. plot:: ae/pf/src/pf_mdl.py
   :include-source: True
   :show-source-link: False
```

We use the `module_printer` to generate two independent python modules `powerflow` and `powerflow_njit` with the `jit` flag being `True` and `False` respectively. 

After *printing* the modules, we can just import these modules and call the `F` and `J` functions to evaluate the power flow model and its Jacobian.

```python
from powerflow import mdl as pf, y as y0
F0 = pf.F(y0, mdl.p)
J0 = pf.J(y0, mdl.p)
```

### Jit acceleration

The above power models have the $\sum$ symbols, which bring about burdensome for-loops. As a potent way to eliminate the for-loops and accelerate calculations, we ought to use the llvm-based Numba package to fully take advantage of the `SMID` in CPUs. It should be noted that Solverz can print numerical codes compatible for Numba integration by setting `jit=True`. 

We show the computation overhead between two `jit` settings using the following figures.

```{eval-rst}
.. plot:: ae/pf/src/time_prof.py
   :show-source-link: True
```

It is apparent that though it took hundreds of seconds to compile the module `powerflow`, the post-compiled `F` and `J` function evaluations are one magnitude faster than those without jit-compilation. 

The compiled results are cached locally, so that only one compilation is required for each model. We recommend that one debug one's models without jit and compile the models in efficiency-demanding cases.

## Ill-conditioned Power flow

The Newton method sometimes fails because it is not robust enough. We view this cases as having ill-conditioned initial settings. In this cases, we can use some more robust methods, such as the semi-implicit continuous Newton method (SICNM)[^sicnm] provided by Solverz. Shown below is an illustrative example of ill-conditioned power flow. The Newton failed while the SICNM easily converged. 


```{eval-rst}
.. plot:: ae/pf/src/ill_pf.py
   :include-source: True
   :show-source-link: False
```

By the way, our implementation of SICNM for MATPOWER can be found [here](https://github.com/rzyu45/MATPOWER-SICNM/blob/main/src/sicnm.m)

[^book1]: F. Milano, Power System Modelling and Scripting, Springer Berlin Heidelberg, 2010. doi: 10.1007/978-3-642-13669-6.
[^sicnm]: R. Yu, W. Gu, S. Lu, and Y. Xu, “Semi-implicit continuous newton method for power flow analysis,” 2023, arXiv:2312.02809.
