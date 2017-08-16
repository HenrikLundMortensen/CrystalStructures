import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

import numpy as np



class plotSurfaceFig:
    """
    Class for plotting surfaces in various configurations. Generally, a matplotlib figure instance is created. 
    The surface is then plotted by updating existing obejcts in the figure. 

    To plot a single surface initialize figure with the initializeSurfacePlot method. 
    To plot a surface, call the plotSurface method. 

    """
    
    def __init__(self):

        # Create the matplotlib figure instance
        self.fig = plt.figure()
        self.masterAx = self.fig.gca()
    

    def initializeSurfacePlot(self,Na):
        """
        Initializes the figure to plot a single surface. Creates patches.Circle instances for each atom.
        The position, color and radius of the circle is then updated later
        
        Input:
        Na: Number of atoms
        """

        ax = self.masterAx

        # Set background color and x and y limits
        ax.set_facecolor((0.9,0.9,1))
        ax.set_xlim([0-0.2,1.2])
        ax.set_ylim([-0.2,1.2])
        ax.set_xticks([])
        ax.set_yticks([])

        # Create patches.Circle instances
        p = []
        for i in range(Na):
            p.append(patches.Circle(xy=(0,0),radius=0))
            ax.add_patch(p[i])
        
        self.patches = p
        
    def plotSurface(self,cs):
        """
        Plots surface by updating the existing patches.Circle instances in self. 

        Input:
        cs: Coordinate set class instance
        """
        # coordinates = cs.coordinates
        coordinates = cs
        
        # Number of atom species
        Na_species = len(set(list(coord[2] for coord in coordinates)))

        radius = 0.05
        colorlist = ['red','blue','black','magenta']

        npcoordinates = np.array(coordinates)
        xmax = max(np.transpose(npcoordinates)[0])
        ymax = max(np.transpose(npcoordinates)[1])                         
                         
        i = 0
        for coords in coordinates:

            p = self.patches[i]
            p.center = (coords[0]/xmax,coords[1]/ymax)
            p.radius = radius

            p.set_color(colorlist[int(coords[2])])
            i += 1

        # Pause to update the graphics
        plt.pause(0.001)
            
if __name__ == '__main__':
    
    Na = 20

    coordinates = np.random.rand(Na,2)
    species = np.random.randint(0,2,size = (Na,1))

    coordinates = np.append(coordinates,species,axis=1)


    surfFig = plotSurfaceFig()
    surfFig.initializeSurfacePlot(Na)

    for i in range(100):
        coordinates = np.random.rand(Na,2)
        species = np.random.randint(0,2,size = (Na,1))
        coordinates = np.append(coordinates,species,axis=1)
        
        surfFig.plotSurface(coordinates)
        plt.pause(0.001)

    surfFig.fig.show()
