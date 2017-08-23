import numpy as np
import crystalStructures.featureVector as fv
import crystalStructures.coordinateSet as cs
import crystalStructures.energyCalculations.energyLennardJones as elj
from sklearn.neural_network import MLPRegressor


def generateData(particles, dataSets):
    size = particles
    N = dataSets
    epsilon, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = [epsilon, r0, sigma]
    energyCalculator = elj.totalEnergyLJdoubleWell

    energyList = []
    featureVectorList = []
    
    # Create large data set
    for i in range(N):
        # Create a new set
        myCoordinateSet = cs.CoordinateSet()
        myCoordinateSet.createRandomSet(size)
        myCoordinateSet.calculateEnergy(energyCalculator, params)
        myCoordinateSet.calculateFeatures(fv.calculateFeatureVectorGaussian)
        # Save energy and feature vectors
        energyList.append(myCoordinateSet.Energy)
        tempFeatureList = []
        for i in range(particles):
            featureVector = myCoordinateSet.FeatureVectors[i]
            if i == 0:
                tempFeatureList = featureVector
            else:
                tempFeatureList = np.concatenate((tempFeatureList, featureVector))
        featureVectorList.append(tempFeatureList)
        
    return np.array(featureVectorList), np.asarray(energyList)

    
if __name__ == '__main__':
    particles, dataSets = 2, 3
    FeatureVectors, EnergyList = generateData(particles, dataSets)

    # Preprocess the feature vectors
    FeatureVectors -= np.mean(FeatureVectors, axis=0)  # Center around zero in each dimension
    FeatureVectors /= np.std(FeatureVectors, axis=0)   # Normalize so each feature is approximately same order

    # Now train the model
    myANN = MLPRegressor()
    myANN.fit(FeatureVectors, EnergyList)

    
    # Generate some test data
    dataSets = 10
