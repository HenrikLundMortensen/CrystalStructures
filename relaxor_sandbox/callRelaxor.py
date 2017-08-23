import numpy as np
import sys
import relaxorBH as rlx

N = int(sys.argv[1])

# Define parameters for energy expression
i_params = int(sys.argv[2])
params_file = np.loadtxt(fname='params.txt', delimiter='\t')
eps, r0, sigma = params_file[i_params, :]
params = (eps, r0, sigma)

# Define boxSize
boxSize = 2*np.sqrt(N)*r0

# random 2D coordinates
x = np.random.rand(N, 2) * boxSize

# Run and time basinhopping
relax = rlx.relaxor(x, rlx.E_LJ_jac, params, boxSize)
relax.runRelaxor()
relax.plotResults()
xres = np.reshape(relax.res.x, (N, 2))

np.savetxt('output' + str(i_params) + '.dat', xres, delimiter='\t')
