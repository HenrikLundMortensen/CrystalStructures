import numpy as np
import crystalStructures.featureVector as fv
import crystalStructures.coordinateSet as cs
import crystalStructures.energyCalculations.energyLennardJones as elj
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as ms
import random


def generateData(particles, dataSets):
    ''' Note that this method only accepts new sets if they are in a certain
    energy range. You have to specify this range yourself. '''
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
        while True:
            myCoordinateSet.createRandomSet(size)
            myCoordinateSet.calculateEnergy(energyCalculator, params)
            if 100 > myCoordinateSet.Energy > 0:
                break
            
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


def generateData1D(dataSets):
    N = dataSets
    particles = 2
    epsilon, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = [epsilon, r0, sigma]
    energyCalculator = elj.totalEnergyLJdoubleWell

    energyList = []
    featureVectorList = []
    
    # Create large data set
    for i in range(N):
        # Create a new set
        myCoordinateSet = cs.CoordinateSet()
        myCoordinateSet.createRandomSet(particles)   # All data sets are two particles
        myCoordinateSet.Coordinates[0][0] = 0        # with some minimum separation
        myCoordinateSet.Coordinates[0][1] = 0
        myCoordinateSet.Coordinates[1][0] = 1 + random.uniform(0, 3)
        myCoordinateSet.Coordinates[1][1] = 0
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
    particles, dataSets = 3, 10
    FeatureVectors, EnergyList = generateData(particles, dataSets)
    print(FeatureVectors)
    # Preprocess the data
    scaler = StandardScaler()
    scaler.fit(FeatureVectors)
    FeatureVectors = scaler.transform(FeatureVectors)
    
    # Now create the model
#    myANN = MLPRegressor(hidden_layer_sizes=(5, 5), max_iter=1000, solver='lbfgs', activation='relu', alpha=0.00001, learning_rate_init=0.01)

#    The below can be used to find optimal parameters for the MLPRegressor.
#    parameters = {'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes': [(10, 10, 10), (30,), (5, 5), (3,)], 'solver': ['adam', 'lbfgs'], 'learning_rate_init': [0.001, 0.01, 0.0001]}
#    gscv = ms.GridSearchCV(myANN, param_grid=parameters)
#    gscv.fit(FeatureVectors, EnergyList)
#    print('Best parameters are:', gscv.best_params_)

    '''
    # Train the model
    myANN.fit(FeatureVectors, EnergyList)
    
    # Check predictions on training data first
    EnergyListPredict = myANN.predict(FeatureVectors)
    error = np.dot(EnergyList - EnergyListPredict, EnergyList - EnergyListPredict) / dataSets
    print('Looking at training data')
    print('Average energy is:', np.average(EnergyList))
    print('The error is:', error)
    print('Error relative to average energy is:', error / np.average(EnergyList), '\n')
    
    # Generate some test data
    particles, dataSets = 3, 10000
    FeatureVectors, EnergyList = generateData(particles, dataSets)
    
    # Now preprocess the new feature vectors
    FeatureVectors = scaler.transform(FeatureVectors)
    
    # Predict energy of new data sets
    EnergyListPredict = myANN.predict(FeatureVectors)
    error = np.dot(EnergyList - EnergyListPredict, EnergyList - EnergyListPredict) / dataSets
    print('Looking at test data')
    print('Average energy is:', np.average(EnergyList))
    print('The error is:', error)
    print('Error relative to average energy is:', error / np.average(EnergyList))
    print('Three predicted energies are:', EnergyListPredict[:10])
    print('Actual energies are:', EnergyList[:10])
    '''
