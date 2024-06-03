(burger)=

# The Analytical Solution to the Inviscid Burger's Equation

The inviscid burger's equation is a representative of the nonlinear hyperbolic partial differential equations (PDEs) with formula
```{math}
\frac{\partial u}{\partial t}+\frac{\partial}{\partial x}\left(\frac{u^2}{2}\right)=0.
```

We illustrate its solution using the initial condition

$$u(x,0)=-\sin(\pi x)$$

and boundary condition

$$u(-1,t)=0\quad u(1,t)=0.$$

The analytical solution can be derived using the method of characteristics. 

First, rewrite the PDE as 

$$\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}=0$$

Then the left hand side can be rewritten as the total derivative of $u(x,t)$, that is,

$$\frac{\mathrm{d}u}{\mathrm{d}t}=\frac{\partial u}{\partial t}+\frac{\partial x}{\partial t}\frac{\partial u}{\partial x}=0.$$

Hence $u$ is constant along the characteristics 

$$\frac{\partial x}{\partial t}=u.$$

Given $u_0$ on $(x_0,0)$, we have, along the characteristic,

$$x_0- u_0\cdot 0=x-u_0t=x+\sin(\pi x_0)t.$$

Therefore, any $u(x,t)$ can be derived by first solving the equation

$$x_0=x+\sin(\pi x_0)t,$$

for $x_0$ and then

$$u(x,t)=-\sin(\pi x_0).$$

```{note}
It should be noted that the solutions of nonlinear hyperbolic PDEs are typically spatially discontinuous.

In this example, a shock formulates at

$$t_b=-\frac{1}{\min_{x\in R}u'_0(x)}=-\frac{1}{\min_{x\in R}-\pi\cos(\pi x)}=\frac{1}{\pi}\approx0.31831$$

and

$$x_s=\frac{1}{2}(u_L+u_R)=0,$$ 

since the initial condition is symmetric about $x=0$.
```

The following codes illustrate the evolution of the shock.
```{literalinclude} src/plot_burger.py
```
Finally, we have
```{eval-rst}
.. plot:: ae/burger/src/plot_burger.py
```
