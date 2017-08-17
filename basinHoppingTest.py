import numpy as np
from scipy.optimize import basinhopping
from energyCalculations.energyLennardJones import totalEnergyLJdoubleWell as LJenergy
from plotSurface import *
from coordinateSet import *
from matplotlib import pyplot as plt
import time





class MyBounds(object):
    def __init__(self, xmax=[1,1,1,1,1,1,1,1], xmin=[-1,-1,-1,-1,-1,-1,-1,-1] ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin



def boundingFunc(coord):
    """

    """

    c = 0
    
    if coord[0]<0:
        c +=  abs(coord[0])# +np.sin(0.5*(abs(coord[0]))*np.pi)
        
    if coord[0]>10:
        c += coord[0]-10# +np.sin(0.5*(coord[0]-10)*np.pi)

    if coord[1]<0:
        c +=  abs(coord[1])# +np.sin(0.5*(abs(coord[1]))*np.pi)
        
    if coord[1]>10:
        c += coord[1]-10# +np.sin(0.5*(coord[1]-10)*np.pi)
    return c




def myfunc(coord):
    """

    """
    x = coord[0]
    y = coord[1]
    return (x-2)**2-3 + (x*y-8)**2-3

def myLJfunc(coordlist):
    """

    """
    coordlist = coordlist.reshape(int(len(coordlist)/2),2)



    E = 0

    
    params = [1.8,1.1,np.sqrt(0.02)]

    E += LJenergy(coordlist,params)

    for coord in coordlist:
        E += boundingFunc(coord)

    return E


Na = 20
cs = CoordinateSet()
cs.createRandomSet(Na)

coordlistold = np.array(cs.Coordinates).reshape(3,Na)[0:2].reshape(Na,2)
coordlist = coordlistold.reshape(1,coordlistold.shape[0]*coordlistold.shape[1])[0]

# clist = []
# for i in range(-10,10):
#     for j in range(-10,10):
#         clist.append(i)
#         clist.append(j)


    

# # mybounds = MyBounds()

t1 = time.time()
minimizer_kwargs = {"method": "BFGS"}
ret = basinhopping(myLJfunc,coordlist,minimizer_kwargs=minimizer_kwargs,stepsize=0.01,T=0.1,niter=200)
t2 = time.time()-t1
print(t2)
print(ret.fun)
print(ret.message)



surfFig = plotSurfaceFig()
surfFig.initializeSurfacePlot(Na)


coords = ret.x.reshape(int(len(ret.x)/2),2)
species = np.random.randint(0,2,size = (Na,1))

coords = np.append(coords,species,axis=1)

# surfFig.plotSurface(ret.x.reshape)
# surfFig.fig.show()
