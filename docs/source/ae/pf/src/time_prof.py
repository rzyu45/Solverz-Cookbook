import matplotlib.pyplot as plt
import time


start = time.perf_counter()
from powerflow import mdl as pf, y as y0
end = time.perf_counter()
compilation_time = end - start

from powerflow_njit import mdl as pf1, y as y1

# %%

start = time.perf_counter()
pf.F(y0, pf.p)
end = time.perf_counter()
Fjit = end - start

start = time.perf_counter()
pf.J(y0, pf.p)
end = time.perf_counter()
Jjit = end - start

start = time.perf_counter()
pf1.F(y0, pf.p)
end = time.perf_counter()
Fnjit = end - start

start = time.perf_counter()
pf1.J(y0, pf.p)
end = time.perf_counter()
Jnjit = end - start
# %% visualize
plt.scatter([1], compilation_time, c='green')
plt.xticks([1], ['Compilation time'])
plt.ylabel('Time/s')
plt.grid()
plt.show()

x = [1, 2]
categories = ['jit', 'njit']
plt.scatter(x[0], Fjit, c='orange', label='F eval')
plt.scatter(x[1], Fnjit, c='orange')
plt.scatter(x[0], Jjit, c='purple', label='J eval')
plt.scatter(x[1], Jnjit, c='purple')
plt.yscale('log')
plt.ylim(1e-5, 1e-1)
plt.xlim([0, 3])
plt.xticks([1, 2], ['jit', 'Non-jit'])
plt.legend()
plt.ylabel('Time/s')
plt.grid()
plt.show()
