import numpy as np
import scipy as sp
from scipy.optimize import basinhopping
import time


def E_LJ_jac(x, *params):
    """
    Calculates total energy and gradient of N atoms interacting with a
    double Lennard-Johnes potential.
    
    Input:
    x: positions of atoms in form x= [x1,y2,x2,y2,...]
    params: parameters for the Lennard-Johnes potential

    Output:
    E: Total energy
    dE: gradient of total energy
    """
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


class takeStep(object):
    """
    Transforms the coordinates into new random positions inside square box
    with sidelrngth "boxSize"
    """
    def __init__(self, boxSize):
        self.boxSize = boxSize

    def __call__(self, x):
        N = np.size(x, 0)
        x = np.random.rand(N) * self.boxSize
        return x


def print_fun(x, f, accepted):
    print("at minimum %.4f accepted %d" % (f, int(accepted)))


class relaxor:
    """
    Class for performing basinhopping and plotting the results.
    
    Input:
    x0: Start positions of the atoms
    Efunc: function that returns total energy and gradient of the energy
    params: parameters for the energy expression
    boxSize: The sidelength of the square box that the atoms are confined in

    Output:
    self.res.x: optimum positions
    self.res.fun: optimum energy
    self.runtime: runtime of the basinhopping algorithm
    
    Methods:
    runRelaxor: Perform the basinhopping optimixation.
    plotResults: Plot the optimum structure.
    """
    def __init__(self, x0, Efunc, params, boxSize):
        self.x = x0
        self.Efunc = Efunc
        self.params = params
        self.boxSize = boxSize

    def print_fun(x, f, accepted):
        print("at minimum %.4f accepted %d" % (f, int(accepted)))
        
    def runRelaxor(self):
        myTakeStep = takeStep(self.boxSize)
        minimizer_kwargs = {"args": self.params, "jac": True}
        t0 = time.time()
        self.res = basinhopping(self.Efunc, self.x,
                                niter=300,
                                take_step=myTakeStep,
                                niter_success=5,
                                minimizer_kwargs=minimizer_kwargs,
                                callback=print_fun)
        self.runtime = time.time() - t0

if __name__ == "__main__":

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

    # Run and time basinhopping
    relax = relaxor(x, E_LJ_jac, params, boxSize)
    relax.runRelaxor()
    xres = relax.res.x
    print("\nRuntime:", relax.runtime)
