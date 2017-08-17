import numpy as np
import crystalStructures.featureVector as fv
import crystalStructures.coordinateGenerator as cg
import crystalStructures.energyCalculations.energyLennardJones as elj


class CoordinateSet:
    """
    #### Attributes ####

    Energy: the energy of the structure. Initially set to zero.

    Coordinates: set of coordinates for all atoms in the strucuture.
                 First element is the x coordinate, second is the y
                 coordinate and third is the atom number.

    FeatureVectorCalculator: instance of the FeatureVector class

    EnergyCalculator: instance of the EnergyCalculator class

    #### Methods ####
    
    createRandomSet: creates a random set of coordinates of a given size

    calculateEnergy: calculates the energy of the surface, given a specific energy calculator

    calculateFeatures: calculates the feature vectors for a GRID/ONE ATOM????
    """

    def __init__(self):
        self.Energy = 0
        self.Coordinates = 0
        self.FeatureVectors = 0

        # Different classes for calculation purposes
        self.FeatureVectorCalculator = fv.FeatureVectorCalculator()

    def createRandomSet(self, size):  # SHOULD PUT RESTRICTIONS ON COORDINATES? MOVE TO DATAGENERATOR
        """ Create a random set of atoms, all of the same kind """
        self.Coordinates = [None] * size
        for coordinate in range(size):
            self.Coordinates[coordinate] = np.random.rand(3)
            self.Coordinates[coordinate][2] = 1

    def calculateEnergy(self, energyCalculator, params):
        self.Energy = energyCalculator(self.Coordinates, params)

    def calculateFeatures(self):
        self.FeatureVectors = self.FeatureVectorCalculator.calculateFeatureVectorsGaussian(self.Coordinates)

            
if __name__ == '__main__':
    size = 2
    myCoordinateSet = CoordinateSet()
    myCoordinateSet.createRandomSet(size)
    print('Coordinates are:', myCoordinateSet.Coordinates)
    myCoordinateSet.calculateFeatures()
    print('Feature vectors are:', myCoordinateSet.FeatureVectors)

    # Calculate energy
    epsilon, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    params = [epsilon, r0, sigma]
    energyCalculator = elj.totalEnergyLJdoubleWell
    myCoordinateSet.calculateEnergy(energyCalculator, params)
    
   
