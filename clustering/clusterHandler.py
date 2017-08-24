from sklearn.cluster import KMeans
import numpy as np


class ClusterHandler:
    """
    Takes a CoordinateSet class with calculated feature vectors
    Then cluster the feature vectors into a given number of clusters
    and returns the KMeans

    #### Attributes ####
    CoordinateSet: in instance of the CoordinateSet class
    K: the number of clusters
    FeatureVectors: the feature vectors one wish to cluster
    KMeans: an instance of KMeans from sklearn.cluster
    
    #### Methods ####
    __init__ : takes a coordinateSet and the desired number of clusters
    doClustering: cluster the given coordinateSet
    """

    def __init__(self, clusters, coordinateSet=None):
        self.CoordinateSet = coordinateSet
        self.K = clusters
        if coordinateSet:
            self.FeatureVectors = coordinateSet.FeatureVectors

    def doClustering(self):
        if isinstance(self.FeatureVectors[0], int):  # Special for 1d feature vectors
            self.Kmeans = KMeans(n_clusters=self.K).fit(np.asarray(self.FeatureVectors).reshape(-1, 1))
        else:
            featVec = np.vstack(self.FeatureVectors)  # Convert into numpy array
            self.Kmeans = KMeans(n_clusters=self.K).fit(featVec)

    def doClusteringList(self, featureVectorList):
        if isinstance(featureVectorList[0], int):  # Special for 1d feature vectors
            self.Kmeans = KMeans(n_clusters=self.K).fit(np.asarray(self.FeatureVectorList).reshape(-1, 1))
        else:
            self.Kmeans = KMeans(n_clusters=self.K).fit(featureVectorList)


if __name__ == '__main__':
    FeatureVectorList = np.array(([0, 0, 0], [1, 1, 1], [2, 2, 2]))
    myClusterHandler = ClusterHandler(3)
    myClusterHandler.doClusteringList(FeatureVectorList)
    print(myClusterHandler.Kmeans.labels_)
    
