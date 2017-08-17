import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping


class globalMinimizer:

    def __init__(self, coorSet, Efunc, params):
        (self.Natoms, self.dim) = np.shape(coorSet)
        self.x0 = np.reshape(coorSet, (self.Natoms * self.dim))
        self.Efunc = Efunc
        self.params = params

    def calcE(self, x):
        return self.Efunc(x, self.params)
        
    def basinhopping(self):
        minimizer_kwargs = {"method": "BFGS", "args": self.params}
        res = basinhopping(self.Efunc, self.x0, stepsize=0.001,
                           minimizer_kwargs=minimizer_kwargs, niter=10)
        return res


def Ecalculator(x, *params):
    eps, r0, sigma = params
    E = 0
    Natoms = np.size(x, 0)
    for i in range(Natoms - 1):
        for j in range(i + 1, Natoms):
            r = np.sqrt(sp.dot(x[i] - x[j], x[i] - x[j]))
            E1 = 1 / np.power(r, 12) - 2 / np.power(r, 6)
            E2 = -eps * np.exp(-np.power(r - r0, 2) / (2 * sigma * sigma))
            E += E1 + E2
    return E
            

N = 3
x = np.random.rand(N, 2) * 20
eps = 1.8
r0 = 1.1
sigma = 0.1
params = (eps, r0, sigma)
minimizer = globalMinimizer(x, Ecalculator, params)
res = minimizer.basinhopping()
xres = np.reshape(res.x, (N, 2))
print(res.x)
plt.plot(x[:, 0], x[:, 1], 'o', color='red')
plt.plot(xres[:, 0], xres[:, 1], 'o', color='blue')
plt.show()
