import numpy as np
import types


class FeatureVectorCalculator:
    """
    Class designed to calculate feature vectors in different ways

    #### Attributes ####

    Coordinates: set of coordiates for all atoms in a structure. Taken from CoordinateSet.
    
    Rc: cutoff radius for atom interaction. Implemented so there is a smooth transition towards this distance.

    #### Methods ####
    
    calculateFeatureVectors: calculate the feature vectors for an entire grid. Default method is
                             simply counting atoms within the cutoff radius. Can provide a custom
                             function for calculating the feature vector.

    calculateSingleFeatureVector: calculate a single feature vector. Takes a grid and the poisition
                                  in the grid

    cutOffFunction: a smooth function that approaches zero at Rc

    #### Methods outside class ####

    calculateFeatureVectorGaussian: calculates a single feature vector in a more advanced way.
                                    Takes a grid and a position.
    """

    def __init__(self, func=None):
        self.Coordinates = 0
        self.Rc = 0.5  # Perhaps take cutoff radius as parameter?
        if func:
            self.calculateSingleFeatureVector = types.MethodType(func, self)

    def calculateFeatureVectors(self, coordinateSet):
        dataSize = len(coordinateSet)
        FeatureVectors = [None] * dataSize
        for i in range(dataSize):
            featureVector = self.calculateSingleFeatureVector(coordinateSet, i)
            FeatureVectors[i] = featureVector
        return FeatureVectors

    def calculateSingleFeatureVector(self, coordinateSet, i):
        dataSize = len(coordinateSet)
        x0, y0 = coordinateSet[i][:2]
        N = 0
        for i in range(dataSize):
            x = coordinateSet[i][0]
            y = coordinateSet[i][1]
            dist = np.sqrt((x0 - x)**2 + (y0 - y)**2)
            if dist < self.Rc:
                N += 1
        return N

    def cutOffFunction(self, x):
        if x <= self.Rc:
            return 0.5 * (1 + np.cos(np.pi * x / self.Rc))
        return 0


def calculateFeatureVectorGaussian(self, coordinateSet, i):
    ''' Calculates the feature vector based on Gaussian approach. Initially just a 2D vector
    with one radial and one angular component. Need to figure out a way to decide on parameters.
    These are just manually set at the moment.'''
        
    featureVector = np.zeros(2)   # Set size of feature vector here
    dataSize = len(coordinateSet)
        
    eta, Rs = 1, 1                # Guess some parameters for radial part
    x0, y0 = coordinateSet[i][:2]
    f1 = 0
    for j in range(dataSize):     # Calculate radial part
        if j == i:
            continue
        x, y = coordinateSet[j][:2]
        Rij = np.sqrt((x0 - x)**2 + (y0 - y)**2)
        if Rij <= self.Rc:
            f1 += np.exp(- eta * (Rij - Rs)**2 / self.Rc * self.Rc) * self.cutOffFunction(Rij)

    xi, lamb, eta = 2, 1, 1    # Guess some parameters for angular part
    f2 = 0                     # Calculate angular part
    for j in range(dataSize):
        if j == i:
            continue
        for k in range(j, dataSize):
            if k == j or k == i:
                continue
            # Calculate the distances between atoms
            RijVec = coordinateSet[j][:2] - coordinateSet[i][:2]
            Rij = np.linalg.norm(RijVec)

            RikVec = coordinateSet[k][:2] - coordinateSet[i][:2]
            Rik = np.linalg.norm(RikVec)

            RjkVec = coordinateSet[k][:2] - coordinateSet[j][:2]
            Rjk = np.linalg.norm(RjkVec)

            f2 += (1 + lamb * np.cos(np.dot(RijVec, RikVec) / (Rij * Rik)))**xi * np.exp(- (Rij * Rij + Rik * Rik + Rjk * Rjk) / self.Rc**2) * self.cutOffFunction(Rij) * self.cutOffFunction(Rik) * self.cutOffFunction(Rjk)
    f2 *= 2**(1 - xi)

    # Set and return feature vector
    featureVector[0] = f1
    featureVector[1] = f2
    return featureVector
