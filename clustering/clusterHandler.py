from sklearn.cluster import KMeans


class ClusterHandler:
    """
    Explanation goes here
    """

    def __init__(self, coordinateSet, clusters):
        self.CoordinateSet = coordinateSet
        self.K = clusters
        self.FeatureVectors = coordinateSet.FeatureVectors

    def doClustering(self):
        self.Kmeans = KMeans(n_clusters=self.K).fit(self.FeatureVectors)

        
if __name__ == '__main__':
    import crystalStructures.coordinateSet as cs
    import matplotlib.pyplot as plt
    import numpy as np
    size = 10

    # Create coordinate set
    myCoordinateSet = cs.CoordinateSet()
    myCoordinateSet.createRandomSet(size)
    myCoordinateSet.calculateFeatures()

    # Create cluster handler
    clusters = 5
    myClusterHandler = ClusterHandler(myCoordinateSet, clusters)
    myClusterHandler.doClustering()
    xData = yData = np.zeros(size)
    for i in range(size):
        data = myClusterHandler.FeatureVectors[i]
        xData[i] = data[0]
        yData[i] = data[1]
    for i in range(size):
        color = myClusterHandler.Kmeans.labels_[i]
        plt.plot(xData[i], yData[i], 'o', color='C' + str(color))
    plt.show()
    
