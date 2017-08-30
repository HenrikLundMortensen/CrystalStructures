import numpy as np
import time
import sys


def loaddata(Ndata, Natoms, Ndim):
    X = np.zeros((Ndata, Natoms, Ndim))
    E = np.zeros(Ndata)

    Ndata_added = 0
    k = 0
    while Ndata_added < Ndata:
        # Load data from datafile number k
        Xadd = np.loadtxt(fname='data40/positions' + str(k) + '.dat', delimiter='\t')
        Eadd = np.loadtxt(fname='data40/energies' + str(k) + '.dat', delimiter='\t')

        # Determine how much of the data in the file is needed to reach Ndata
        Ndata_in_file = np.size(Eadd, 0)
        Ndata2add = min(Ndata_in_file, Ndata - Ndata_added)

        # Add data from file to the previously loaded data
        # + add the position data as 3D array
        for i in range(Ndata2add):
            X[Ndata_added+i, :, :] = Xadd[:, 2*i:2*i+2]
        E[Ndata_added:Ndata_added + Ndata2add] = Eadd[0:Ndata2add]

        # Count current amount of data loaded and iterate
        Ndata_added += Ndata2add
        k += 1
    return X, E


def loaddatafile(Ndata, Natoms, Ndim, i_file):
    
    # Load data from datafile number k
    X = np.loadtxt(fname='data50/positions' + str(i_file) + '.dat', delimiter='\t')
    E = np.loadtxt(fname='data50/energies' + str(i_file) + '.dat', delimiter='\t')

    X = blockshaped(X, Natoms, Ndim)
    return X, E


def cutOffFunction(x, Rc):
        if x <= Rc:
            return 0.5 * (1 + np.cos(np.pi * x / Rc))
        else:
            return 0


def getFeatures(X, Rc, params_2body, params_3body):
    (Ndata, Natoms, Ndim) = np.shape(X)
    N2D_params = np.size(params_2body, 0)
    N3D_params = np.size(params_3body, 0)
    Nfeatures = N2D_params + N3D_params
    Xf = np.zeros((Ndata, Natoms, Nfeatures))
    for n in range(Ndata):
        for i in range(Natoms):
            for s in range(N2D_params):
                eta = params_2body[s, 0]
                Rs = params_2body[s, 1]
                f2d = 0
                for j in range(Natoms):
                    if j == i:
                        continue
                    Rij = np.linalg.norm(X[n, j, :]-X[n, i, :])
                    if Rij < Rc:
                        f2d += np.exp(- eta * (Rij - Rs)**2 / Rc**2) * cutOffFunction(Rij, Rc)
                Xf[n, i, s] = f2d
            for p in range(N3D_params):
                xi = params_3body[p, 0]
                lamb = params_3body[p, 1]
                eta = params_3body[p, 2]
                f3d = 0
                for j in range(Natoms):
                    if j == i:
                        continue
                    for k in range(j, Natoms):
                        if k == j or k == i:
                            continue
                        RijVec = X[n, j, :] - X[n, i, :]
                        Rij = np.linalg.norm(RijVec)
                        
                        RikVec = X[n, k, :] - X[n, i, :]
                        Rik = np.linalg.norm(RikVec)
                        
                        RjkVec = X[n, k, :] - X[n, j, :]
                        Rjk = np.linalg.norm(RjkVec)
                        
                        f3d += (1 + lamb * np.dot(RijVec, RikVec) / (Rij * Rik))**xi * np.exp(- eta * (Rij**2 + Rik**2 + Rjk**2) / Rc**2) * cutOffFunction(Rij, Rc) * cutOffFunction(Rik, Rc) * cutOffFunction(Rjk, Rc)
                Xf[n, i, N2D_params + p] = f3d
    return Xf


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1, 2)
               .reshape(h, w))

    
if __name__ == "__main__":
    
    Ndata = 100
    Natoms = 50
    Ndim = 2

    X, E = loaddatafile(Ndata, Natoms, Ndim, sys.argv[1])
    print("shape(X):", np.shape(X))
    Nfeatures = 5
    Rc = 3
    params_2body = np.array([[2, 0], [8, 0], [40, 0]])
    params_3body = np.array([[2, 1, 0.005], [2, -1, 0.005]])
    t0 = time.time()
    Xf = getFeatures(X, Rc, params_2body, params_3body)
    print("1. shape(Xf):", np.shape(Xf))
    for i in range(np.size(Xf, 2)):
        Xf[:, :, i] = (Xf[:, :, i] - np.mean(Xf[:, :, i])) / np.std(Xf[:, :, i])
    print("2. shape(Xf):", np.shape(Xf))
    Xf = unblockshaped(Xf, Natoms, Ndata*Nfeatures)
    runtime = time.time() - t0
    print("Runtime: ", runtime)
        
    np.savetxt('features' + str(sys.argv[1]) + '.dat', Xf, delimiter='\t')
