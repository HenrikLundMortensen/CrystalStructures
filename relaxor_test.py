import numpy as np
import scipy as sp
from scipy.optimize import basinhopping
import time
import sys


def E_LJ(x, *params):
    eps, r0, sigma = params
    E = 0
    N = np.size(x, 0)
    x = np.reshape(x, (int(N / 2), 2))
    Natoms = np.size(x, 0)
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
    dE = np.zeros(Natoms)
    for i in range(N):
        for j in range(i + 1, Natoms):
            r = np.sqrt(sp.dot(x[i] - x[j], x[i] - x[j]))
            E1 = 1 / np.power(r, 12) - 2 / np.power(r, 6)
            E2 = -eps * np.exp(-np.power(r - r0, 2) / (2*sigma**2))
            E += E1 + E2
            if j != i:
                dxij = x[2*i] - x[2*j]
                dyij = x[2*i + 1] - x[2*j + 1]
                
                dEx1 = 12*dxij*(-1 / r**10 + 1 / r**4)
                dEx2 = eps*(r - r0)*r*dxij * np.exp(-np.power(r - r0, 2) / (2*sigma**2))
                dEy1 = 12*dyij*(-1 / r**10 + 1 / r**4)
                dEy2 = eps*(r - r0)*r*dyij * np.exp(-np.power(r - r0, 2) / (2*sigma**2))

                dE[2*i] += dEx1 + dEx2
                dE[2*i + 1] += dEy1 + dEy2
    return E, dE


def myTakeStep(x):
    boxSize = 6
    N = np.size(x, 0)
    x = np.random.rand(N) * boxSize
    return x


# Make random atom configuration
N = 4
boxSize = 6
x = np.random.rand(N, 2) * boxSize  # for plotting
x0 = np.reshape(x, N * 2)  # Reshape for basinhopping

# Define parameters for energy expression
i_params = int(sys.argv[1])
params_file = np.loadtxt(fname='params.txt', delimiter='\t')
eps, r0, sigma = params_file[i_params, :]
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
xres = np.reshape(xres, (N, 2))
print(xres)
# print final positions to file
np.savetxt('output' + str(i_params) + '.dat', xres, delimiter='\t')
