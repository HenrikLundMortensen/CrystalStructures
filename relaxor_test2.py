import numpy as np
import scipy as sp
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
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
#eps = 1.8  # 1.8
#r0 = 1.1  # 1.1
#sigma = np.sqrt(0.02)
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

# print final positions to file
np.savetxt('output' + str(i_params) + '.dat', xres, delimiter='\t')

'''
# Plot box
Xbox = [0, boxSize, boxSize, 0, 0]
Ybox = [0, 0, boxSize, boxSize, 0]
plt.plot(Xbox, Ybox, color='black')

# Plot atoms
xres = np.reshape(xres, (N, 2))  # Reshape for plotting 
plt.plot(xres[:, 0], xres[:, 1], 'o', color='red', ms=2)
#sideSpace = 0.1
#plt.xlim([-sideSpace * boxSize, (1 + sideSpace) * boxSize])
#plt.ylim([-sideSpace * boxSize, (1 + sideSpace) * boxSize])
plt.show()
'''
