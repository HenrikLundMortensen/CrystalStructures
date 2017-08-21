import numpy as np
import scipy.linalg as linalg
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


def createEnergyModel(CNmatrix, Elist):
    """
    Input:
    CNmatrix: Cluster number matrix
    Elist: Energy list

    Output:
    EC: Energy associated with each cluster
    """

    EC = np.dot(linalg.pinv(CNmatrix), Elist)
    return EC


def getEnergyFromModel(CN, EC):
    """
    Input:
    CN: Number of each clusters
    EC: Energy associated with each cluster

    Output:
    Emodel: Energy predicted by the model
    """

    Emodel = np.dot(CN, EC)
    return Emodel


if __name__ == '__main__':
    import crystalStructures.featureVector as fv
    import crystalStructures.coordinateSet as cs
    import crystalStructures.energyCalculations.energyLennardJones as elj
    import crystalStructures.clustering.clusterHandler as ch
    
    size = 2
    N = 100
    epsilon, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = [epsilon, r0, sigma]
    energyCalculator = elj.totalEnergyLJdoubleWell

    mainCoordinateSet = cs.CoordinateSet()  # Main coordinate set used for clustering
    energyList = []
    featureVectorList = []
    
    mainCoordinateSet.createRandomSet(size)
    mainCoordinateSet.calculateEnergy(energyCalculator, params)
    mainCoordinateSet.calculateFeatures(fv.calculateFeatureVectorGaussian)
    energyList.append(mainCoordinateSet.Energy)
    featureVectorList.append(mainCoordinateSet.FeatureVectors)

    # Create large data set
    for i in range(N):
        # Create a new set
        myCoordinateSet = cs.CoordinateSet()
        myCoordinateSet.createRandomSet(size)
        myCoordinateSet.calculateEnergy(energyCalculator, params)
        myCoordinateSet.calculateFeatures(fv.calculateFeatureVectorGaussian)

        # Append to old set and save energy and feature vectors
        mainCoordinateSet.Coordinates.append(myCoordinateSet.Coordinates)
        mainCoordinateSet.FeatureVectors.append(myCoordinateSet.FeatureVectors)
        energyList.append(myCoordinateSet.Energy)
        featureVectorList.append(myCoordinateSet.FeatureVectors)

    # Now do the clustering using the big set
    K = 5
    mainCoordinateSet.calculateFeatures(fv.calculateFeatureVectorGaussian)
    myClusterHandler = ch.ClusterHandler(mainCoordinateSet, K)
    myClusterHandler.doClustering()

    # Now for each of the feature vector sets, predict which clusters they are in
    clusterList = []
    for featureVector in featureVectorList:
        tempCoordinateSet = cs.CoordinateSet()
        tempCoordinateSet.FeatureVectors = featureVector
        tempCoordinateSet.calculateClusters(myClusterHandler.Kmeans)
        clusterList.append(tempCoordinateSet.Clusters)

    EModel = createEnergyModel(np.asarray(clusterList), np.asarray(energyList))

    # Now create some test data
    N = 10
    EnergyList = []
    ClusterList = []
    for i in range(N):
        # Create a new set
        myCoordinateSet = cs.CoordinateSet()
        myCoordinateSet.createRandomSet(size)
        myCoordinateSet.calculateEnergy(energyCalculator, params)
        myCoordinateSet.calculateFeatures(fv.calculateFeatureVectorGaussian)

        # Predict the clusters in the set
        myCoordinateSet.calculateClusters(myClusterHandler.Kmeans)

        # Save the new data
        ClusterList.append(myCoordinateSet.Clusters)
        EnergyList.append(myCoordinateSet.Energy)
        
    # Now predict energies of the test set
    EnergyListPredict = []
    for i in range(N):
        EnergyListPredict.append(getEnergyFromModel(np.asarray(ClusterList[i]), EModel))

    # Calculate energy difference
    error = np.dot(np.asarray(EnergyListPredict) - np.asarray(EnergyList), np.asarray(EnergyListPredict) - np.asarray(EnergyList)) / N

    print('The average error is:', error)
