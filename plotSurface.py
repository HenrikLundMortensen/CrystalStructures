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

        # Create patches.Circle instances
        p = []
        for i in range(Na):
            p.append(patches.Circle(xy=(0,0),radius=0))
            ax.add_patch(p[i])

        # Create rectangle to define boundary box
        bb = patches.Polygon([[0,0],[0,0],[0,0],[0,0]],lw=0,fill=False,color='black')
        ax.add_patch(bb)

        self.bb = bb
        self.patches = p
        
    def plotSurface(self,cs,**kwargs):
        """
        Plots surface by updating the existing patches.Circle instances in self. 

        Input:
        cs: Coordinate set class instance
        """


        if kwargs is not None:
            for key,value in kwargs.items():
                if key == "bounds":
                    self.bb.xy = [[value.xmin,value.ymin],
                                  [value.xmin,value.ymax],
                                  [value.xmax,value.ymax],
                                  [value.xmax,value.ymin]]
                    self.bb.set_linewidth(2)
                    
        # Coordinates = cs.Coordinates
        Coordinates = cs
        
        # Number of atom species
        Na_species = len(set(list(coord[2] for coord in Coordinates)))


        colorlist = ['red','blue','black','magenta']

        npCoordinates = np.array(Coordinates)

        xmax = max(np.transpose(npCoordinates)[0])
        xmin = min(np.transpose(npCoordinates)[0])
        ymax = max(np.transpose(npCoordinates)[1])
        ymin = min(np.transpose(npCoordinates)[1])


        self.masterAx.set_xlim([xmin-abs(xmin)*0.2,xmax+abs(xmax)*0.2])
        self.masterAx.set_ylim([ymin-abs(ymin)*0.2,ymax+abs(ymax)*0.2])
        self.masterAx.set_aspect('equal')
        
        radius = 0.05
                         
        i = 0
        for coords in Coordinates:

            p = self.patches[i]
            p.center = (coords[0],coords[1])
            p.radius = radius
            p.set_alpha(0.9)

            p.set_color(colorlist[int(coords[2])])
            i += 1

            


            
        # Pause to update the graphics
        plt.pause(0.001)
            
if __name__ == '__main__':
    
    Na = 20

    Coordinates = np.random.rand(Na,2)
    species = np.random.randint(0,2,size = (Na,1))

    Coordinates = np.append(Coordinates,species,axis=1)


    surfFig = plotSurfaceFig()
    surfFig.initializeSurfacePlot(Na)

    for i in range(100):
        Coordinates = np.random.rand(Na,2)
        species = np.random.randint(0,2,size = (Na,1))
        Coordinates = np.append(Coordinates,species,axis=1)
        
        surfFig.plotSurface(Coordinates)
        plt.pause(0.001)

    surfFig.fig.show()
