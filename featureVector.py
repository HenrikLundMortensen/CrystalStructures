import numpy as np


class FeatureVectorCalculator:
    """
    Description goes here
    """

    def __init__(self):
        self.Coordinates = 0
        self.Rc = 0.5  # Perhaps take cutoff radius as parameter?

    def calculateFeatureVectorsSimple(self, coordinateSet):
        dataSize = len(coordinateSet)
        FeatureVectors = [None] * dataSize
        for i in range(dataSize):
            featureVector = self.calculateSingleFeatureVectorSimple(coordinateSet, i)
            FeatureVectors[i] = featureVector
        return FeatureVectors

    def calculateSingleFeatureVectorSimple(self, coordinateSet, i):
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

    def calculateFeatureVectorsGaussian(self, coordinateSet):
        dataSize = len(coordinateSet)
        FeatureVectors = [None] * dataSize
        for i in range(dataSize):
            featureVector = self.calculateSingleFeatureVectorGaussian(coordinateSet, i)
            FeatureVectors[i] = featureVector
        return FeatureVectors

    def calculateSingleFeatureVectorGaussian(self, coordinateSet, i):
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
        for j in range(dataSize):  # Counts all triples twice, not sure if intended
            if j == i:
                continue
            for k in range(dataSize):
                if k == j or k == i:
                    continue
                # Calculate the distances between atoms
                RijVec = coordinateSet[j][:2] - coordinateSet[i][:2]
                Rij = np.linalg.norm(RijVec)

                RikVec = coordinateSet[k][:2] - coordinateSet[i][:2]
                Rik = np.linalg.norm(RikVec)

                RjkVec = coordinateSet[k][:2] - coordinateSet[j][:2]
                Rjk = np.linalg.norm(RjkVec)
                f2 += (1 + lamb * np.cos(np.dot(RijVec, RikVec) / (Rij * Rik)))**xi * np.exp(- (Rij * Rij + Rik * Rik + Rjk * Rjk) /
                                                                                             self.Rc**2) * self.cutOffFunction(Rij) * self.cutOffFunction(Rik) * self.cutOffFunction(Rjk)
        f2 *= 2**(1 - xi)

        # Set and return feature vector
        featureVector[0] = f1
        featureVector[1] = f2
        return featureVector
        
    def cutOffFunction(self, x):
        if x <= self.Rc:
            return 0.5 * (1 + np.cos(np.pi * x / self.Rc))
        return 0
                
                
                
            
            
            

