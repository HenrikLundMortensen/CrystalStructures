import numpy as np
from matplotlib import pyplot as plt


def dist(coord1,coord2):
    """
    Calculates and returns Euclidian distance between coordinates
    """
    return np.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)


def LJdoubleWell(r,params):
    """
    Double well Lennard Jones potential for a pair of atoms

    Input:
    r: Atom pair seperation
    params: Parameters, [epsilon, r0, sigma]

    Output:
    Value of the potential at r
    """
    epsilon = params[0]
    r0 = params[1]
    sigma = params[2]

    # Return double well Lennard Jones potential energy value at r
    return 1/(r**12) - 2/(r**6) - epsilon*np.exp( -  ((r - r0)**2)/(2*sigma**2))


def totalEnergyLJdoubleWell(coords,params):
    """
    Calculates the total double well Lennard Jones potential energy of the atom ensemble defined by the given coordinate set.

    Input:
    coords: Coordinates of atoms
    params: Parameters for the potential, [epsilon, r0, sigma] 

    Output:
    E: Total energy

    """

    E = 0

    # Calculate energy pair wise.  
    for i in range(len(coords)):
        for j in range(i+1,len(coords)): # Does not calculate for the same pairs twice.
            r = dist(coords[i],coords[j])
            E += LJdoubleWell(r,params)
    return E



if __name__ == '__main__':
    
    a = 0
    params = [1.8,1.1,np.sqrt(0.02)]
    print(totalEnergyLJdoubleWell(a,params))


    # Reproduce the two examples in fig. 5 of "Detecting crystal structures..."-article
    r = np.linspace(0.8,3,100)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(r,LJdoubleWell(r,[1.8,1.1,np.sqrt(0.02)]))
    ax.plot(r,LJdoubleWell(r,[1.8,1.9,np.sqrt(0.02)]))

    ax.set_ylabel('V(r)')
    ax.set_xlabel('r')
    ax.set_ylim([-3,2])
    
    fig.savefig('LJplot.png')
