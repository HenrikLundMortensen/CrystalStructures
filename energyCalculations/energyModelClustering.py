import numpy as np
from sklearn.neural_network import MLPRegressor
import sklearn.model_selection as ms
import crystalStructures.featureVector as fv
import crystalStructures.coordinateSet as cs
import crystalStructures.energyCalculations.energyLennardJones as elj
import crystalStructures.clustering.clusterHandler as ch


def generateData(particles, dataSets):
    ''' Note that this method only accepts new sets if they are in a certain
    energy range. You have to specify this range yourself. Also the parameters of the
    Lennard Jones potential must be specified'''
    size = particles
    N = dataSets
    epsilon, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = [epsilon, r0, sigma]
    energyCalculator = elj.totalEnergyLJdoubleWell

    energyList = []
    dataSetList = []
    
    # Create the data sets
    for i in range(N):
        # Create a new set
        myCoordinateSet = cs.CoordinateSet()
        while True:
            myCoordinateSet.createRandomSet(size)
            myCoordinateSet.calculateEnergy(energyCalculator, params)
            if 100 > myCoordinateSet.Energy > 0:
                break
            
        energyList.append(myCoordinateSet.Energy)
        dataSetList.append(np.vstack(myCoordinateSet.Coordinates))
        
    return dataSetList, np.array(energyList)


def clusterLocalData(dataSets, K):
    ''' Must change feature vector calculation if needed '''
    FeatureVectors = []

    # First group all the feature vectors
    for i in range(len(dataSets)):
        tempCoordinateSet = cs.CoordinateSet()
        tempCoordinateSet.Coordinates = dataSets[i]
        tempCoordinateSet.calculateFeatures(fv.calculateFeatureVectorGaussian)
        FeatureVectors += tempCoordinateSet.FeatureVectors

    # Then do the clustering
    myClusterHandler = ch.ClusterHandler(K)
    myClusterHandler.doClusteringList(FeatureVectors)

    return myClusterHandler.Kmeans


def predictLocalCluster(dataSet, KMeans):
    ''' Must change feature vector calculation if needed '''
    coordinateSet = cs.CoordinateSet()
    coordinateSet.Coordinates = dataSet
    coordinateSet.calculateFeatures(fv.calculateFeatureVectorGaussian)
    coordinateSet.calculateClusters(KMeans)
    return coordinateSet.Clusters


if __name__ == '__main__':

    # Create data
    particles, N = 5, 100
    dataSets, energyList = generateData(particles, N)

    # Find local clusters using feature vectors from all datasets
    K = 5
    KMeans = clusterLocalData(dataSets, K)

    # Now create global feature vector for each data set
    featureVectorList = np.zeros((N, K))
    for i in range(N):
        clusters = predictLocalCluster(dataSets[i], KMeans)
        featureVectorList[i] = clusters

    # Now with feature vectors and energies try to create an ANN
    myANN = MLPRegressor(hidden_layer_sizes=(5, 5), max_iter=3000, solver='lbfgs', activation='relu', alpha=0.00001, learning_rate_init=0.01)

#    The below can be used to find optimal parameters for the MLPRegressor.
    parameters = {'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes': [(1, 1), (10, 10, 10), (30,), (5, 5), (3,)], 'solver': ['adam', 'lbfgs'], 'learning_rate_init': [0.001, 0.01, 0.0001]}
    gscv = ms.GridSearchCV(myANN, param_grid=parameters)
    gscv.fit(featureVectorList, energyList)
    print('Best parameters are:', gscv.best_params_)

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
    
