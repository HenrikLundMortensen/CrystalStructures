import numpy as np
import scipy as sp
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
import time


def E_LJ(x, *params):
    eps, r0, sigma = params
    E = 0
    N = np.size(x, 0)
    x = np.reshape(x, (int(N/2), 2))
    Natoms = np.size(x, 0)
    for i in range(Natoms):
        for j in range(i + 1, Natoms):
            r = np.sqrt(sp.dot(x[i] - x[j], x[i] - x[j]))
            E1 = 1 / np.power(r, 12) - 2 / np.power(r, 6)
            E2 = -eps * np.exp(-np.power(r - r0, 2) / (2 * sigma * sigma))
            E += E1 + E2
    return E


class MyBounds(object):
    def __init__(self, xmax=[0, 0, 0, 0, 0, 0], xmin=[1, 1, 1, 1, 1, 1]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

def mybounds(**kwargs):
    x = kwargs["x_new"]
    tmax = bool(np.all(x <= 1.0))
    tmin = bool(np.all(x >= 0.0))
    print(tmax and tmin)
    return tmax and tmin


def myTakeStep(x):
    boxSize = 17
    N = np.size(x, 0)
    x = np.random.rand(N) * boxSize
    return x

# Make random atom configuration
N = 60
boxSize = 17
x = np.random.rand(N, 2) * boxSize  # for plotting
x0 = np.reshape(x, N*2)  # Reshape for basinhopping

# Define parameters for energy expression
eps = 1  # 1.8
r0 = 1.4  # 1.1
sigma = np.sqrt(0.02)
params = (eps, r0, sigma)
minimizer_kwargs = {"args": params}

# Run and time basinhopping
t0 = time.time()
res = basinhopping(E_LJ, x, niter=300, take_step=myTakeStep,
                   niter_success=15, minimizer_kwargs=minimizer_kwargs)
print("time:", time.time() - t0)
print(res.message)
# Extract optimum positions
xres = res.x
xres = np.reshape(xres, (N, 2))  # Reshape for plotting

# Plot box
Xbox = [0, boxSize, boxSize, 0, 0]
Ybox = [0, 0, boxSize, boxSize, 0]
plt.plot(Xbox, Ybox, color='black')
# Plot atoms

plt.plot(xres[:, 0], xres[:, 1], 'o', color='red', ms=2)
#sideSpace = 0.1
#plt.xlim([-sideSpace * boxSize, (1 + sideSpace) * boxSize])
#plt.ylim([-sideSpace * boxSize, (1 + sideSpace) * boxSize])
plt.show()
