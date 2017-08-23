import numpy as np
import scipy as sp
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
import time


def E_LJ(x, *params):
    eps, r0, sigma = params
    N = np.size(x, 0)
    x = np.reshape(x, (int(N / 2), 2))
    Natoms = np.size(x, 0)

    E = 0
    for i in range(Natoms):
        for j in range(i + 1, Natoms):
            r = np.sqrt(sp.dot(x[i] - x[j], x[i] - x[j]))
            E1 = 1 / np.power(r, 12) - 2 / np.power(r, 6)
            E2 = -eps * np.exp(-np.power(r - r0, 2) / (2 * sigma * sigma))
            E += E1 + E2
    return E


def E_LJ_jac(x, *params):
    eps, r0, sigma = params
    N = np.size(x, 0)
    x = np.reshape(x, (int(N / 2), 2))
    Natoms = np.size(x, 0)

    E = 0
    dE = np.zeros(N)
    for i in range(Natoms):
        for j in range(Natoms):
            r = np.sqrt(sp.dot(x[i] - x[j], x[i] - x[j]))
            if j > i:
                E1 = 1/r**12 - 2/r**6
                E2 = -eps * np.exp(-(r - r0)**2 / (2*sigma**2))
                E += E1 + E2
            if j != i:
                dxij = x[i, 0] - x[j, 0]
                dyij = x[i, 1] - x[j, 1]

                dEx1 = 12*dxij*(-1 / r**14 + 1 / r**8)
                dEx2 = eps*(r-r0)*dxij / (r*sigma**2) * np.exp(-(r - r0)**2 / (2*sigma**2))

                dEy1 = 12*dyij*(-1 / r**14 + 1 / r**8)
                dEy2 = eps*(r-r0)*dyij / (r*sigma**2) * np.exp(-(r - r0)**2 / (2*sigma**2))

                dE[2*i] += dEx1 + dEx2
                dE[2*i + 1] += dEy1 + dEy2
    return E, dE


def myTakeStep(x):
    boxSize = 6
    N = np.size(x, 0)
    x = np.random.rand(N) * boxSize
    return x


def print_fun(x, f, accepted):
    print("at minimum %.4f accepted %d" % (f, int(accepted)))


# Make random atom configuration
N = 10
boxSize = 6
x = np.random.rand(N, 2) * boxSize  # for plotting
x0 = np.reshape(x, N * 2)  # Reshape for basinhopping

# Define parameters for energy expression
eps = 1.8  # 1.8
r0 = 1.1  # 1.1
sigma = np.sqrt(0.02)
params = (eps, r0, sigma)

# Run and time basinhopping with analytic gradient
minimizer_kwargs = {"args": params, "jac": True}
t0 = time.time()
res = basinhopping(E_LJ_jac, x, niter=300, take_step=myTakeStep,
                   niter_success=5, minimizer_kwargs=minimizer_kwargs,
                   callback=print_fun)
print("\ntime:", time.time() - t0)
print("\n", res.message)
xres = res.x

# compare optimum energies for the two runs
E_anl_opt, dE = E_LJ_jac(xres, eps, r0, sigma)
print("\nOptimum Energy:", E_anl_opt)

# Plot box
Xbox = [0, boxSize, boxSize, 0, 0]
Ybox = [0, 0, boxSize, boxSize, 0]
plt.plot(Xbox, Ybox, color='black')

# Plot atoms
xres = np.reshape(xres, (N, 2))  # Reshape for plotting
plt.plot(xres[:, 0], xres[:, 1], 'o', color='red', ms=2)
# sideSpace = 0.1
# plt.xlim([-sideSpace * boxSize, (1 + sideSpace) * boxSize])
# plt.ylim([-sideSpace * boxSize, (1 + sideSpace) * boxSize])
plt.show()

"""
class tkStep(object):
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize

    def __call__(self, x):
        s = self.stepsize
        x += np.random.uniform(-s, s, x.shape)
        xmin, xmax = 0, 6
        above = bool(np.all(x >= xmax))
        below = bool(np.all(x <= xmin))
        if above or below:
            self.nstep += 1
            
        return x

   
k = np.random.rand(10) * 2 - 0.5
hlow = k < 0
hhigh = k > 1
print(np.c_[k, k*hlow, k*hhigh])
"""
