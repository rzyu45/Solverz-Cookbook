(pf)=

# The Calculation of Electric Power Flow

*Author: [Ruizhi Yu](https://github.com/rzyu45)*

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

where $p_h$ and $q_h$ are respectively the active and reactive injection power of bus $h$; $v_h$ is the voltage magnitude of bus $h$; $\theta_{hk}=\theta_h-\theta_k$ is the voltage angle difference between bus $h$ and $k$; $g_{hk}$ and $b_{hk}$ are the $(h, k)$-th entry of the conductance and susceptance matrices; $\mathbb{B}$ is the set of bus indices with the subscripts being the bus types.

The buses in electric power systems are typically sparsely connected, and hence the Jacobian of power flow models are always sparse. In what follows, we will set the `sparse` flag to be `True`.

## Implementation in Solverz

### Power flow modelling

We use the `case30` from the [matpower](https://matpower.org/) library. The required data for verification can be found in case file directory of the [source repo](https://github.com/rzyu45/Solverz-Cookbook).

We first perform the symbolic modelling of the `case30` power flow. 

```{literalinclude} src/pf_mdl.py
```

We use the `module_printer` to generate two independent python modules `powerflow` and `powerflow_njit` with the `jit` flag being `True` and `False` respectively. 

After *printing* the modules, we can just import these modules and call the `F` and `J` functions to evaluate the power flow model and its Jacobian.

```python
from powerflow import mdl as pf, y as y0
F0 = pf.F(y0, pf.p)
J0 = pf.J(y0, pf.p)
```

### Jit acceleration

The above power models have the $\sum$ symbols, which bring about burdensome for-loops. As a potent way to eliminate the for-loops and accelerate calculations, we ought to use the llvm-based Numba package to fully take advantage of the `SMID` in CPUs. It should be noted that Solverz can print numerical codes compatible for Numba integration by setting `jit=True`. 

We show the computation overhead between two `jit` settings using the following figures.

![omega](fig/time_prof.png)

![omega](fig/time_prof_01.png)

On a laptop equipped with Ryzen 5800H CPU, it took hundreds of seconds to compile the module `powerflow`. However, the post-compiled `F` and `J` function evaluations were one magnitude faster than those without jit-compilation. 

The compiled results are cached locally, so that only one compilation is required for each model. We recommend that one debug one's models without jit and compile the models in efficiency-demanding cases.

## Matrix-form power flow with `Mat_Mul`

Since version 0.8.0, Solverz supports a compact matrix-vector formulation of power flow using `Mat_Mul`. Instead of element-wise for-loops, we express the power injection equations in rectangular coordinates ($e$, $f$) using matrix-vector products:

```{math}
\left\{
\begin{aligned}
&\mathbf{p}=\mathbf{e}\odot(\mathbf{G}\mathbf{e}-\mathbf{B}\mathbf{f})+\mathbf{f}\odot(\mathbf{B}\mathbf{e}+\mathbf{G}\mathbf{f})\\
&\mathbf{q}=\mathbf{f}\odot(\mathbf{G}\mathbf{e}-\mathbf{B}\mathbf{f})-\mathbf{e}\odot(\mathbf{B}\mathbf{e}+\mathbf{G}\mathbf{f})
\end{aligned}
\right.
```

where $\mathbf{G}$ and $\mathbf{B}$ are the conductance and susceptance matrices (sparse), and $\odot$ denotes element-wise multiplication.

Solverz automatically computes the symbolic Jacobian via {ref}`matrix calculus <matrix_calculus>`. For example, the Jacobian of the active power equation w.r.t. $\mathbf{e}$ is:

```{math}
\frac{\partial\mathbf{p}}{\partial\mathbf{e}}=\operatorname{diag}(\mathbf{G}\mathbf{e}-\mathbf{B}\mathbf{f})+\operatorname{diag}(\mathbf{e})\mathbf{G}+\operatorname{diag}(\mathbf{f})\mathbf{B}
```

The implementation for the `case30` system:

```{literalinclude} src/pf_matmul.py
```

Starting from a flat start ($e=1$, $f=0$), the Newton-Raphson method converges in 4 iterations.

```{note}
The `Mat_Mul` formulation is much more compact than the for-loop approach — the entire power flow model is defined in just a few lines. The Jacobian is computed automatically by the matrix calculus engine, eliminating the need for manual derivation.
```

## Performance comparison: `Mat_Mul` vs. for-loop

The two formulations above model the same physical system (`case30`) but pay very different costs at different phases of the workflow. The benchmark script is in [`src/bench_pf_matmul_vs_polar.py`](src/bench_pf_matmul_vs_polar.py) and can be re-run on any hardware:

```bash
cd docs/source/ae/pf/src
python bench_pf_matmul_vs_polar.py
```

It times seven phases end-to-end in a single run, with the module cold-compile phase executed in a fresh subprocess (and with `__pycache__` / Numba `.nbi`/`.nbc` caches wiped beforehand) so the "first-time user" compile cost is measured honestly.

Numbers below are from a 2025 MacBook Air (Apple M4), averaged across two consecutive runs:

| Phase                                         |      for-loop (polar) |       Mat_Mul (rect.) | Mat_Mul wins by |
| :-------------------------------------------- | --------------------: | --------------------: | --------------: |
| 1. `Model() → create_instance()`              |              ≈ 2.3 s  |              ≈ 0.06 s |           ~35× |
| 2. `FormJac(y0)`                              |              ≈ 0.06 s |             ≈ 0.006 s |            ~9× |
| 3. Inline compile (`made_numerical`)          |             ≈ 0.31 s  |              ≈ 0.01 s |           ~30× |
| 4. Inline hot **F** (per call)                |              ≈ 175 µs |               ≈ 16 µs |            ~11× |
| 4. Inline hot **J** (per call)                |              ≈ 860 µs |              ≈ 265 µs |            ~3× |
| 5. Module render (`module_printer.render`)    |             ≈ 0.55 s  |              ≈ 0.02 s |           ~33× |
| 6. **Module cold compile** (import + Numba)   |            **≈ 47 s** |            **≈ 2.5 s**|           ~19× |
| 7. Module hot **F** (per call)                |             **≈ 1.4 µs** |            ≈ 14 µs  |    0.1× *(loses)* |
| 7. Module hot **J** (per call)                |               ≈ 66 µs |               ≈ 53 µs |          ~1.3× |

Shapes: the polar form has **53 unknowns / 53 scalar `Eqn`s** (`Va` at PV+PQ buses, `Vm` at PQ buses); the `Mat_Mul` form has **58 unknowns / 3 vector `Eqn`s** (`e`, `f` at non-ref buses, with P balance + Q balance + V² at PV). The comparison is not strictly equi-dimensional but close enough that the differences are driven by the formulation, not the unknown count.

### Compile-time cost

**`Mat_Mul` compiles ~19× faster on a cold import.** This is the headline number and scales directly with the number of `@njit` functions the code generator emits:

- **for-loop (polar)** — 1 dispatcher `inner_F` + **53** per-equation `inner_F{i}` + 1 dispatcher `inner_J` + **361** per-non-zero `inner_J{k}`. That's **~416 Numba kernels** to compile on the first run, each one a scalar trig expression. Each individual kernel is cheap to compile but the fixed overhead per kernel (LLVM instantiation, symbol table, cache write) adds up to ~45 seconds.
- **`Mat_Mul` (rectangular)** — 1 dispatcher `inner_F` + **3** per-vector-equation `inner_F{i}` + 1 dispatcher `inner_J` + a handful of per-block `inner_J{k}` + **4** per-mutable-matrix-block `_mut_block_N` scatter-add kernels. Total: ~10 kernels. Cold compile is dominated by import + Numba startup (~2 s) rather than by per-kernel compilation.

Every earlier phase (model construction, `FormJac`, `made_numerical`, `render`) follows the same 20–40× scaling, because they all traverse the same explosion of scalar equations.

### Runtime cost

Once the modules are compiled, the picture is more nuanced:

- **Module hot F** is one of the few places the for-loop path *wins*, by ~10×. Drilling into what a single `Mat_Mul` `F_` call actually spends its time on:

  | Step | Cost | % of total |
  |---|---:|---:|
  | 8 `scipy.sparse` SpMVs in the wrapper (`G_nr@e`, `B_nr@f`, `B_nr@e`, `G_nr@f`, `G_pq@e`, `B_pq@f`, `B_pq@e`, `G_pq@f`) | ~11.7 µs | **83 %** |
  | `@njit inner_F` call (dispatch + 3 vector equations) | ~1.6 µs | 11 % |
  | 30 dict lookups on `p_` | ~0.6 µs | 4 % |
  | **Total `F_(y, p)`** | **~14.1 µs** | 100 % |

  A single `G_nr @ e` scipy SpMV on case30 (58×58, ~250 nnz) already costs ~1.5 µs — 100 % of which is Python→C dispatch overhead; the actual matvec arithmetic is well under 100 ns. Eight of those SpMVs ≈ 12 µs, and that's what dominates hot F.

  The polar form has no sparse matrices at runtime at all: 53 fully inlined `@njit` scalar kernels collapse into one Python→Numba boundary crossing. **For small networks like case30 the scipy call overhead dominates the per-call budget**; the ratio narrows (or inverts) as the network grows and the actual matvec work starts to dominate.

- **Module hot J** is roughly a tie, with `Mat_Mul` slightly ahead (~1.3×). The `Mat_Mul` Jacobian kernel uses the vectorised scatter-add path described in the {ref}`Matrix-Vector Calculus <matrix_calculus>` chapter, which is O(nnz) per block; the polar path visits 361 scalar `inner_J{k}` functions but each one is a pure computation without sparse overhead. The two paths almost exactly cancel out at this size.
- **Inline hot F/J** — without Numba, the for-loop form is slower across the board (11× on F, 3× on J) because lambdify has to walk 53 large scalar expressions + 361 Jacobian sub-expressions on every call. `Mat_Mul`'s 3 vector equations + scipy SpMVs finish in a fraction of the time.

```{note}
The 30 dict lookups at the top of the `Mat_Mul` `F_` wrapper include 16 **unused** legacy fields (`G_nr_data`, `G_nr_indices`, `G_nr_indptr`, `G_nr_shape0` and the same for every other sparse `dim=2` parameter) — these were emitted by the old `MatVecMul` code path and the current `Mat_Mul` architecture does not use them, but the code generator still loads and forwards them to `inner_F`. Skipping them would recover roughly 0.3 µs per call and about 16 argument slots in the generated signature. Tracked as a follow-up optimisation.
```

### Which formulation should I use?

- **`Mat_Mul` is the right choice for ~every case larger than a toy example.** The 20–40× wins on every compile phase matter much more than the 10× loss on hot F: the compile-time savings are paid on *every model rebuild*, while the hot-F difference (~13 µs) is invisible next to the J call (~50 µs) and the linear solve on anything with more than a few dozen buses.
- **The for-loop form is marginally preferable only when**: (a) you are shipping a compiled model once and running it a very large number of times — millions of F evaluations — so the one-time compile cost is amortised, and (b) the network is small enough that scipy.sparse SpMV overhead dominates over its per-nnz work. Even then, the compile-time gap usually tips the balance back to `Mat_Mul` if you're iterating on the model.
- **For larger networks** (case118 and beyond) the SpMV work starts to dominate the per-call overhead and `Mat_Mul`'s hot F catches up or surpasses the for-loop form — the 10× loss shown here is specific to the case30 scale.

```{note}
The cold-compile cost for the for-loop form (~47 s on M4) matches the "hundreds of seconds" figure quoted for the older Ryzen 5800H laptop earlier in this chapter. The *absolute* number is sensitive to CPU single-thread performance, but the **ratio** between the two formulations (≈19×) is driven almost entirely by the number of `@njit` kernels the code generator emits, which is a property of the formulation, not the hardware.
```

## Ill-conditioned Power flow

The Newton method sometimes fails because it is not robust enough. We view this cases as having ill-conditioned initial settings. In this cases, we can use some more robust methods, such as the semi-implicit continuous Newton method (SICNM)[^sicnm] provided by Solverz. Shown below is an illustrative example of ill-conditioned power flow. The Newton failed while the SICNM easily converged. 


```{literalinclude} src/ill_pf.py
```

![omega](fig/ill_pf.png)

By the way, our implementation of SICNM for MATPOWER can be found [here](https://github.com/rzyu45/MATPOWER-SICNM/blob/main/src/sicnm.m)

[^book1]: F. Milano, Power System Modelling and Scripting, Springer Berlin Heidelberg, 2010. doi: 10.1007/978-3-642-13669-6.
[^sicnm]: R. Yu, W. Gu, S. Lu, and Y. Xu, “Semi-implicit continuous newton method for power flow analysis,” 2023, arXiv:2312.02809.
