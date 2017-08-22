import numpy as np
import matplotlib.pyplot as plt

boxSize = 6
for i in range(10):
    plt.figure(i)
    xload = np.loadtxt(fname='results/output' + str(i) + '.dat', delimiter='\t')
    Xbox = [0, boxSize, boxSize, 0, 0]
    Ybox = [0, 0, boxSize, boxSize, 0] 
    plt.plot(Xbox, Ybox, color='black')

    # Plot atoms
    plt.plot(xload[:, 0], xload[:, 1], 'o', color='red', ms=2)
plt.show()
