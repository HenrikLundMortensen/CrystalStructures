import numpy as np
from scipy.optimize import basinhopping
from energyCalculations.energyLennardJones import totalEnergyLJdoubleWell as LJenergy
from plotSurface import *
from coordinateSet import *
from matplotlib import pyplot as plt
import time
import pickle


class Boundaries():
    """
    This class is for defining a boundary box for the basin hopping algorithm.
    Outside this box the energy gets a contribution according to the function 
    BoundFunc (default is linear).

    The BasinHopping class creates and instance of this class. 

    Attributes:
    xmin: Lower end of the x-axis of the box
    xmax: Higher end of the x-axis of the box
    ymin: Lower end of the y-axis of the box
    ymax: Higher end of the y-axis of the box
    func: The function that determines the energy contribution outside the box

    """
    def defaultFunc(x):
        """
        Linear bounding function as default
        """
        return x

    def __init__(self, xmin,xmax,ymin,ymax,Boundfunc=defaultFunc):
        """
        Initializes.
        """
        
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.func = Boundfunc
        
    def __call__(self,coord):
        """
        When the instance is called, it checks whether the xy-coordinates (coord) is 
        outisde the box. If so, it returns the contribution according to the BoundFunc. 
        """
        c = 0
        x = coord[0]
        y = coord[1]
        
        if x<self.xmin:
            c += self.func(abs(x-self.xmin))
        if x>self.xmax:
            c += self.func(abs(x-self.xmax))
        if y<self.ymin:
            c += self.func(abs(y-self.ymin))
        if y>self.ymax:
            c += self.func(abs(y-self.ymax))
        return c



class TakeStep(object):
    """
    This defines how to take steps in the basin hopping algorithm. The step is taken by
    adding a random number to each coordinate. The random number is chosen in the interval [-stepsize,stepsize].
    If the coordinate is outside the bounding box defined by the Boundaries class instance, a new random number is 
    calculated.
    """
    

    def __init__(self,stepsize,Bounds):
        """
        """
        self.stepsize = stepsize
        self.Bounds = Bounds
        
    def __call__(self,x):
        """
        When the class instance is called, a new set of coordinates are generated and returned.
        
        Input:
        x: coordinate list , [x1,y1,x2,y2,x3,y3,...]

        output:
        x: Perturbed coordinate list , [x1,y1,x2,y2,x3,y3,...]
        """

        s = self.stepsize
        b = self.Bounds

        # Check if x is within the boundary box
        i = 0
        while i < len(x):
            # if i is even, then x[i] is an x-coordinate
            if np.mod(i,2) == 0:
                # if the coordinate is within bounds, move on
                if x[i] < b.xmax and x[i] > b.xmin:
                    i+=1
                else:
                    # Move inside box
                    x[i] = np.random.uniform(b.xmin,b.xmax)
            # if i is odd, then x[i] is an y-coordinate            
            else:
                # if the coordinate is within bounds, move on
                if x[i] < b.ymax and x[i] > b.ymin:
                    i+=1
                else:
                    # Move inside box
                    x[i] = np.random.uniform(b.ymin,b.ymax)


        # Displace
        i = 0        
        while i < len(x):
            xtmp = x[i] + np.random.uniform(-s,s)
            # if i is even, then x[i] is an x-coordinate
            if np.mod(i,2) == 0:
                # if the coordinate is within bounds, move on
                if xtmp < b.xmax and xtmp > b.xmin:
                    x[i] = xtmp
                    i+=1
            # if i is odd, then x[i] is an y-coordinate            
            else:
                # if the coordinate is within bounds, move on
                if xtmp < b.ymax and xtmp > b.ymin:
                    x[i] = xtmp
                    i+=1
        return x


def callBackFunc(x,f,accepted):
    """
    Prints the intermediate energies found by the basin hopping algorithm.
    """
    print("E: %0.4f accepted %d" %(f,int(accepted)))
    
    
    
def foldCoordList(coords):
    """
    Takes a 1x2N numpy array and returns 2xN numpy array

    Input:
    coords: numpy array, [x1,y1,x2,y2,x3,y3,...]
    
    Output:
    numpy array , [[x1,y1],[x2,y2],[x3,y3],...]
    """
    return coords.reshape(int(len(coords)/2),2)



def unfoldCoordList(coords):
    """
    Takes a 2xN numpy array and returns 1x2N numpy array

    Input:
    coords: numpy array , [[x1,y1],[x2,y2],[x3,y3],...]
    
    Output:
    numpy array, [x1,y1,x2,y2,x3,y3,...]
    """
    coordtmp = np.array(coords).reshape(3,len(coords))[0:2].reshape(len(coords),2)
    return coordtmp.reshape(1,coordtmp.shape[0]*coordtmp.shape[1])[0]

def energyFuncWrapper(coords,*args):
    """
    This function mergers the real energy expression with the boundary box

    Input:
    coords: numpy array with coordinates, [x1,y1,x2,y2,x3,y3,...]
    *args: Arguments: [params, Boundaries class instance, energy function]
    """

    # Reshapes coordinates into [[x1,y1],[x2,y2],,...] which the energy function needs
    coords = foldCoordList(coords)
    E = 0

    params = args[0]
    bounds = args[1]
    energyFunc = args[2]

    E += energyFunc(coords,params)

    # for coord in coords:
    #     E += bounds(coord)
        
    return E

class BasinHopping():
    """
    Wrapper of scipy.optimize.basinhopping algorithm. Use as follows;

    Create instance of CoordinateSet class. Define the paratemers of the energy function and 
    put into list. Decide on xmin, xmax, ymin, ymax for the boundary box. Initialize BasinHopping 
    class instance as
    
    BH = BasinHopping(CoordinateSet class instance, 
                      Energy function, 
                      list of parameters,
                      bounds=[xmin,xmax,ymin,ymax])

    Run the basin hopping algorithm with

    BH.runBasinHopping()

    The optimized coordinates is stored as
    BH.optimizedCoords
    """

    def __init__(self,cs,energyFunc,params,bounds=[-10000,10000,-10000,-10000]):
        """
        Initialization. 

        Input:
        cs: CoordinateSet class instance
        energyFunc: Energy function which takes (cs.Coordinates,params) as arguments
        params: Parameters for the energy function
        bounds: list of four numbers determining [xmin, xmax, ymin, ymax] of the boundary box

        """
        self.cs = cs
        self.bounds = Boundaries(bounds[0],bounds[1],bounds[2],bounds[3])
        self.energyFunc = energyFunc
        self.params = tuple(params)

    def runBasinHopping(self):
        """
        Runs the Basin Hopping algotithm

        Adds the following attributes
        
        init_coords: Coordinates before optimization.
        self.ret: Output of scipy.optimize.basinhopping
        self.optimizedCoords: Optimized coordinates of atoms
        """

        self.init_coords = self.cs.Coordinates
        speciesList = np.transpose(self.init_coords)[2]
        unfolded_coords = unfoldCoordList(self.init_coords)

        args = (self.params,) + (self.bounds,self.energyFunc)


        MyTakeStep = TakeStep(0.5,self.bounds)
        minimizer_kwargs = {"method": "BFGS","args":args}

        t1 = time.time()
        ret = basinhopping(energyFuncWrapper,
                           unfolded_coords,
                           minimizer_kwargs=minimizer_kwargs,
                           take_step=MyTakeStep,
                           callback = callBackFunc,
                           stepsize = 0.1,
                           T=1,
                           niter=200,
                           niter_success=5)

        t2 = time.time() - t1
        self.ret = ret
        self.optimizedCoords = np.append(foldCoordList(ret.x),speciesList.reshape(len(speciesList),1),axis=1)
        self.runtime = t2





if __name__ == '__main__':
    
    bounds = Boundaries(0,30,0,30)

    Na = 20
    cs = CoordinateSet()
    cs.createRandomSet(Na)

    params = [1.8,1.1,np.sqrt(0.02)]


    BS = BasinHopping(cs,LJenergy,params,bounds=[0,10,0,10])
    BS.runBasinHopping()
    print(BS.runtime)


    
