import matplotlib.pyplot as plt
import numpy as np


def plot_structure(x, boxSize):
    N = np.size(x, 0)
    x = np.reshape(x, (int(N/2), 2))
    plt.figure(1)
    Xbox = [0, boxSize, boxSize, 0, 0]
    Ybox = [0, 0, boxSize, boxSize, 0]
    plt.plot(Xbox, Ybox, color='black')
    
    # Plot atoms
    plt.plot(x[:, 0], x[:, 1], 'o', color='red', ms=2)
