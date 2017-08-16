import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping


class coordinateGenerator:
    """
    Natoms: number of atoms in the coordinate set.
    dim: The dimention of the spece the atoms are positioned in (dim = 2 for 2D ect.)
    atomContent (not added yet): Array to specify the distribution of each atom type.
    """
    
    def __init__(self, Natoms, dim):
        self.Natoms = Natoms
        self.dim = dim
        
    def genCoordinateSet(self):
        coorSet = []
        for i in range(self.Natoms):
            newAtom = np.r_[np.random.rand(self.dim), 0]
            coorSet.append(newAtom)
        return coorSet
    
    def genDataSet(self, Ndata):
        dataSet = []
        for i in range(Ndata):
            dataSet.append(self.genCoordinateSet())
        return dataSet


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

"""
N = 100

eps = 0.1
r0 = 0.5
sigma = 0.1

r = np.linspace(0, 3, N)
x = np.c_[r, np.zeros(N)]

E_array = np.zeros(N)
for i in range(1, N):
    E_array[i] += Ecalculator(x[(0, i), :], eps, r0, sigma)

plt.plot(r, E_array)
plt.ylim((-2, 1))
plt.show()
"""
