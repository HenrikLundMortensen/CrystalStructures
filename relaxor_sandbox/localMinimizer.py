import numpy as np
from scipy.optimize import minimize
from relaxorBH import E_LJ_jac
import sys


def stretch(x):
    N = np.size(x, 0)
    return np.reshape(x, 2*N)


N = 100
params = (1, 1.4, np.sqrt(0.02))
boxsize = 1.5*np.sqrt(N)

Ndata = 100
k = 0
positions = np.zeros((N, 2*Ndata))
energies = np.zeros(Ndata)
while k < Ndata:
    x0 = np.random.rand(N, 2) * boxsize
    bounds = [(0, boxsize)] * N * 2
    res = minimize(E_LJ_jac, x0, params,
                   method="TNC",
                   jac=True,
                   tol=1e-3,
                   bounds=bounds)
    if res.fun < 0:
        xres = np.reshape(res.x, (N, 2))
        positions[:, 2*k:2*k+2] = xres
        energies[k] = res.fun
        k += 1
    
np.savetxt('positions' + str(sys.argv[1]) + '.dat', positions, delimiter='\t')
np.savetxt('energies' + str(sys.argv[1]) + '.dat', energies, delimiter='\t')
