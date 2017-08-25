import numpy as np
from scipy.optimize import minimize
from relaxorBH import E_LJ_jac
import matplotlib.pyplot as plt
import time
import sys


def stretch(x):
    N = np.size(x, 0)
    return np.reshape(x, 2*N)


N = 10
params = (1, 1.4, np.sqrt(0.02))
boxsize = 1.5*np.sqrt(N)

Ndata = 4
k = 0
results = np.zeros((N, 2*Ndata))
while k < 2*Ndata:
    x0 = np.random.rand(N, 2) * boxsize
    bounds = [(0, boxsize)] * N * 2
    res = minimize(E_LJ_jac, x0, params,
                   method="TNC",
                   jac=True,
                   tol=1e-3,
                   bounds=bounds)
    if res.fun < 0:
        xres = np.reshape(res.x, (N, 2))
        results[:, k:k+2] = xres
        k += 2
    
np.savetxt('output' + str(sys.argv[1]) + '.dat', results, delimiter='\t')

"""
#xres = np.reshape(res.x, (N, 2))

#E0, dE0 = E_LJ_jac(stretch(x0), params[0], params[1], params[2])
#Eres, dEres = E_LJ_jac(stretch(xres), params[0], params[1], params[2])


#print("boxsize:", boxsize)
#print("E0:", E0)
#print("Eres:", Eres)

# Plot box
Xbox = [0, boxsize, boxsize, 0, 0]
Ybox = [0, 0, boxsize, boxsize, 0]
plt.plot(Xbox, Ybox, color='black')

plt.plot(x0[:, 0], x0[:, 1], 'o', color='red', ms=2)
plt.plot(xres[:, 0], xres[:, 1], 'o', color='blue', ms=2)
plt.show()
"""
