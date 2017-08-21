import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
import energyCalculations.energyLennardJones as elj


class globalMinimizer:

    def __init__(self, coorSet, Efunc, params):
        (self.Natoms, self.dim) = np.shape(coorSet)
        self.x0 = np.reshape(coorSet, (self.Natoms * self.dim))
        self.Efunc = Efunc
        self.params = params

    def calcE(self, x):
        return self.Efunc(x, self.params)
        
    def basinhopping(self):
        minimizer_kwargs = {"args": self.params}
        res = basinhopping(self.Efunc, self.x0,
                           minimizer_kwargs=minimizer_kwargs, niter=100)
        return res


def Ecalculator(x, params):
    eps = params[0]
    r0 = params[1]
    sigma = params[2]
    E = 0
    Natoms = np.size(x, 0)
    for i in range(Natoms):
        for j in range(i + 1, Natoms):
            r = np.sqrt(sp.dot(x[i] - x[j], x[i] - x[j]))
            E1 = 1 / np.power(r, 12) - 2 / np.power(r, 6)
            E2 = -eps * np.exp(-np.power(r - r0, 2) / (2 * sigma * sigma))
            E += E1 + E2
    return E
            

"""
eps = 1.8
r0 = 1.1
sigma = 0.1
params = [eps, r0, sigma]
r = np.linspace(0.8,3,100)
fig = plt.figure()
ax = fig.gca()
ax.plot(r,Ecalculator(r,params))
ax.plot(r,Ecalculator(r,params))

ax.set_ylabel('V(r)')
ax.set_xlabel('r')
ax.set_ylim([-3,2])
plt.show()

N = 2
x = np.random.rand(N, 2) * 4
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
"""


class MyBounds(object):
    
    def __init__(self, xmax=[0, 0, 0, 0, 0, 0, 0, 0], xmin=[2, 2, 2, 2, 2, 2, 2, 2]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
        
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

    
N = 4
eps = 1.8
r0 = 1.1
sigma = np.sqrt(0.02)
params = [eps, r0, sigma]
x0 = np.random.rand(N, 2) * 1
x0 = np.reshape(x0, 2 * N)
print(np.shape(x0))
minimizer_kwargs = {"args": params}
res = basinhopping(Ecalculator, x0, niter=200, minimizer_kwargs=minimizer_kwargs, niter_success=5, accept_test=MyBounds)
xres = np.reshape(res.x, (N, 2))
plt.plot(xres[:, 0], xres[:, 1], 'o', color='blue')
plt.show()
print(res)
