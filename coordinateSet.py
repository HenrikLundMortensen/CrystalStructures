import numpy as np
import crystalStructures.featureVector as fv


class CoordinateSet:
    """
    #### Attributes ####

    Energy: the energy of the structure. Initially set to zero.

    Coordinates: set of coordinates for all atoms in the strucuture.
                 First element is the x coordinate, second is the y
                 coordinate and third is the atom number.

    FeatureVectorCalculator: instance of the FeatureVector class

    EnergyCalculator: instance of the EnergyCalculator class

    #### Methods ####
    
    createRandomSet: creates a random set of coordinates of a given size

    calculateEnergy: calculates the energy of the surface, given a specific energy calculator

    calculateFeatures: calculates the feature vectors of the atoms

    calculateClusters: calculate the clusters in the surface sample. Takes a KMean
    """

    def __init__(self):
        self.Energy = 0
        self.Coordinates = 0
        self.FeatureVectors = 0

    def createRandomSet(self, size):  # SHOULD PUT RESTRICTIONS ON COORDINATES? MOVE TO DATAGENERATOR
        """ Create a random set of atoms, all of the same kind """
        self.Coordinates = [None] * size
        
        for i in range(size):
            self.Coordinates[i] = np.random.rand(3) * 20
            self.Coordinates[i][2] = 1

    def calculateEnergy(self, energyCalculator, params):
        self.Energy = energyCalculator(self.Coordinates, params)

    def calculateFeatures(self, func=None):
        self.FeatureVectorCalculator = fv.FeatureVectorCalculator(func)
        self.FeatureVectors = self.FeatureVectorCalculator.calculateFeatureVectors(self.Coordinates)

    def calculateClusters(self, KMeans):
        if isinstance(self.FeatureVectors[0], int):
            clusterList = KMeans.predict(np.asarray(self.FeatureVectors).reshape(-1, 1))
        else:
            featVec = np.vstack(self.FeatureVectors)
            clusterList = KMeans.predict(featVec)
        K = len(KMeans.cluster_centers_)  # Number of clusters
        self.Clusters = np.bincount(clusterList, minlength=K)

        
if __name__ == '__main__':
    size = 5
    # Create random set and calculate feature vectors
    myCoordinateSet = CoordinateSet()
    myCoordinateSet.createRandomSet(size)
    print('Coordinates are:', myCoordinateSet.Coordinates)
    myCoordinateSet.calculateFeatures(fv.calculateFeatureVectorGaussian)
    print('Feature vectors are:', myCoordinateSet.FeatureVectors)

    
    # Calculate energy
    import crystalStructures.energyCalculations.energyLennardJones as elj
    epsilon, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = [epsilon, r0, sigma]
    energyCalculator = elj.totalEnergyLJdoubleWell
    myCoordinateSet.calculateEnergy(energyCalculator, params)
    print('Energy is:', myCoordinateSet.Energy)

    # Cluster the feature vectors
    import crystalStructures.clustering.clusterHandler as ch
    myClusterHandler = ch.ClusterHandler(myCoordinateSet, 3)
    myClusterHandler.doClustering()

    # Count the atoms in each cluster in the set
    myCoordinateSet.calculateClusters(myClusterHandler.Kmeans)
    print(myCoordinateSet.Clusters)

    
    
   
