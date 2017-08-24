import numpy as np
import matplotlib.pyplot as plt

N = 100
params_file = np.loadtxt(fname='params.txt', delimiter='\t')
for i in range(10):
    eps, r0, sigma = params_file[i, :]

    # Define boxSize
    boxSize = 1.5*np.sqrt(N)*r0

    plt.figure(i)
    xload = np.loadtxt(fname='results100/output' + str(i) + '.dat', delimiter='\t')
    Xbox = [0, boxSize, boxSize, 0, 0]
    Ybox = [0, 0, boxSize, boxSize, 0] 
    plt.plot(Xbox, Ybox, color='black')

    # Plot atoms
    plt.plot(xload[:, 0], xload[:, 1], 'o', color='red', ms=2)
plt.show()
