import numpy as np
from plotSurface import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('coordFile')
args = parser.parse_args()



coordinates = np.loadtxt(args.coordFile,delimiter='\t')

surfFig = plotSurfaceFig()
surfFig.initializeSurfacePlot(len(coordinates))
surfFig.plotSurface(coordinates)
surfFig.fig.savefig(''.join([args.coordFile,'_figure.png']))
