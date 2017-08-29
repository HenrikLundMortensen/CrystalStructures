import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
import time


class Regressor(BaseEstimator):
    def __init__(self, Nk=0.01):
        self.Nk = Nk

    def stretchData(self, X, Ndata, Natoms, Nfeatures):
        return np.reshape(X, (Ndata * Natoms, Nfeatures))
    
    def predict(self, X):
        (Ntest, Natoms, Nfeatures) = X.shape
        X = self.stretchData(X, Ntest, Natoms, Nfeatures)
        X = np.reshape(self.kmeans.predict(X), (Ntest, Natoms))
        Nmat = np.zeros((Ntest, self.Nk)).astype(int)
        for i in range(Ntest):
            Nmat[i, :] = np.bincount(X[i, :], minlength=self.Nk)
        Epredict = np.dot(Nmat, self.Ecluster)
        return Epredict

    def fit(self, X, E, **kwargs):
        self.Nk = kwargs['Nk']
        (Ntrain, Natoms, Nfeatures) = np.shape(X)
        F = self.stretchData(X, Ntrain, Natoms, Nfeatures)

        self.kmeans = KMeans(n_clusters=self.Nk)
        self.kmeans.fit(F)

        # Extract labeled atoms from clustering
        F = np.reshape(self.kmeans.labels_, (Ntrain, Natoms))

        # Calculate Nmat (matrix containing Number of atoms belonging to each cluster for
        # each image.)
        Nmat = np.zeros((Ntrain, self.Nk)).astype(int)
        for i in range(Ntrain):
            Nmat[i, :] = np.bincount(F[i, :], minlength=self.Nk)

        # Calculate cluster energies by solving E = Nmat*Ecluster
        # (in least squares sense using pseudo-inverse)
        self.Ecluster = np.dot(np.linalg.pinv(Nmat), E)
        
    def score(self, Xtest, Etest):
        Ntest = np.size(Etest, 0)
        Epredict = self.predict(Xtest)
        return 1/Ntest * np.dot(Etest - Epredict, Etest - Epredict)

    
def loaddata(Ndata, Natoms, Ndim):
    X = np.zeros((Ndata, Natoms, Ndim))
    E = np.zeros(Ndata)

    Ndata_added = 0
    k = 0
    while Ndata_added < Ndata:
        # Load data from datafile number k
        Xadd = np.loadtxt(fname='data30/features' + str(k) + '.dat', delimiter='\t')
        Eadd = np.loadtxt(fname='data30/energies' + str(k) + '.dat', delimiter='\t')

        # Determine how much of the data in the file is needed to reach Ndata
        Ndata_in_file = np.size(Eadd, 0)
        Ndata2add = min(Ndata_in_file, Ndata - Ndata_added)
        
        # Add data from file to the previously loaded data
        # + add the position data as 3D array
        for i in range(Ndata2add):
            X[Ndata_added+i, :, :] = Xadd[:, Ndim*i:Ndim*(i+1)]
        E[Ndata_added:Ndata_added + Ndata2add] = Eadd[0:Ndata2add]
        
        # Count current amount of data loaded and iterate
        Ndata_added += Ndata2add
        k += 1
    return X, E


if __name__ == '__main__':
    Nfeatures = 5
    Natoms = 30

    Npoints = 20  # Points on learning curve
    Ndata_array = np.logspace(1, 3, Npoints).astype(int)

    error_array = np.zeros(Npoints)
    for i in range(Npoints):
        Ndata = Ndata_array[i]
        X, E = loaddata(Ndata, Natoms, Nfeatures)
        reg = Regressor(Nk=50)

        t0 = time.time()
        error = cross_val_score(reg,
                                X,
                                E,
                                fit_params={'Nk': 50},
                                cv=3)

        error_array[i] = np.mean(error)
        print("Runtime:", time.time() - t0)
        print("error:", error)

LC = np.c_[Ndata_array, error_array]
np.savetxt('LCdata_30atoms_3CV_50Nk.dat', LC, delimiter='\t')
