import crystalStructures.energyCalculations.energyLennardJones as lj
import crystalStructures.coordinateSet as cs
import scipy.optimize.basinhopping as bh

myCoordinateSet = cs.CoordinateSet()
myCoordinateSet.createRandomSet(5)

params = [1, 1, 1]
