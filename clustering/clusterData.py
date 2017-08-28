import crystalStructures.coordinateSet as cs
import crystalStructures.clustering.clusterHandler as ch
import crystalStructures.featureVector as fv
import os
import numpy as np
import matplotlib.pyplot as plt


def parseData():
    # IF LOCATION OF DATA CHANGES, THIS MUST BE CHANGED
    path = '../grendelResults/335820.in1/'
    dataSets = []
    index = []
    params = []

    paramFile = np.loadtxt(os.path.join(path, 'paramList.dat'))

    # Find all the data files and their index
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and 'optimizedCoordinates_' in file:
            fullPath = os.path.join(path, file)
            stringStartIndex = file.find('Index_') + len('Index_')
            stringEndIndex = file.find('_eps')
            fileIndex = file[stringStartIndex:stringEndIndex]
            index.append(fileIndex)                            # save file index
            dataSets.append(np.loadtxt(fullPath))              # save data
            params.append(paramFile[int(fileIndex), :])        # save parameters

    return dataSets, index, params


def clusterLocalData(dataSets):
    FeatureVectors = []

    # First group all the feature vectors
    for i in range(len(dataSets)):
        tempCoordinateSet = cs.CoordinateSet()
        tempCoordinateSet.Coordinates = dataSets[i]
        tempCoordinateSet.calculateFeatures(fv.calculateFeatureVectorGaussian)
        FeatureVectors += tempCoordinateSet.FeatureVectors
    # Then do the clustering
    K = 5  # Number of clusters
    myClusterHandler = ch.ClusterHandler(K)
    myClusterHandler.doClusteringList(FeatureVectors)

    return myClusterHandler.Kmeans


def clusterGlobalData(globalFeatureVectorList):
    K = 3
    myClusterHandler = ch.ClusterHandler(K)
    myClusterHandler.doClusteringList(globalFeatureVectorList)
    return myClusterHandler.Kmeans


def predictLocalCluster(dataSet, KMeans):
    coordinateSet = cs.CoordinateSet()
    coordinateSet.Coordinates = dataSet
    coordinateSet.calculateFeatures(fv.calculateFeatureVectorGaussian)  # Should take function or do this differently
    coordinateSet.calculateClusters(KMeans)
    return coordinateSet.Clusters


def plotPhases(dataSets, labels, params):
    # Find parameters for each grid
    r0 = np.zeros(len(dataSets))
    eps = np.zeros(len(dataSets))
    for i in range(len(dataSets)):
        eps[i] = params[i][0]
        r0[i] = params[i][1]

    # Plot the cluster for each parameter
    for i in range(len(dataSets)):
        color = int(labels[i])
        plt.plot(r0[i], eps[i], 'o', color='C' + str(color))
    plt.show()


def plotClusters(dataSets, labels):
    clusters = np.amax(labels) + 1
    maxPlot = 4  # Maximum number of plots in each cluster
    fig, axarr = plt.subplots(maxPlot, clusters)
    for j in range(clusters):
        k = 0
        for i in range(len(dataSets)):
            clusterType = labels[i]
            if k == maxPlot:
                continue
            if clusterType == j:
                data = dataSets[i]
                xData = data[:, 0]
                yData = data[:, 1]
                axarr[k, j].plot(xData, yData, 'o')
                axarr[k, j].set_xlim([0, 20])
                axarr[k, j].set_ylim([0, 20])
                axarr[k, j].set_xticks([])
                axarr[k, j].set_yticks([])
                k += 1
    plt.show()


if __name__ == '__main__':
    
    dataSets, index, params = parseData()
    KMeans = clusterLocalData(dataSets)
    
    # Calculate a list of global feature vectors
    globalFeatureVectors = []
    for i in range(len(dataSets)):
        clusters = predictLocalCluster(dataSets[i], KMeans)
        globalFeatureVectors.append(clusters)

    # Cluster the global feature vectors
    globalKMeans = clusterGlobalData(globalFeatureVectors)
    labels = globalKMeans.labels_
    print(labels)  # For Grendel

    # Plot the phase diagram
    plotPhases(dataSets, labels, params)

    # Plot the clusters
    plotClusters(dataSets, labels)
    


