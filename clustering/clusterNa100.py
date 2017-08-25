import numpy as np
import glob
from coordinateSet import *
from clusterHandler import *


cslist=[]
m = 0
for coords in glob.glob('/home/henrik/crystalStructures/grendelResults/335820.in1/*.dat'):
    cs = CoordinateSet()
    cs.Coordinates = np.loadtxt(coords)
    cslist.append(cs)


    
chlist = []

for cs in cslist:
    ch = ClusterHandler(10,cs)
    ch.doClustering()
    chlist.append(ch)
    
    
