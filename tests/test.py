import unittest
import numpy as np
# run from the parent directory to solve this import correctly
from src.network import Network

class NetworkTest(unittest.TestCase):

    def testNetworkWith3Layers(self):
        ex = Network(sizes=[2,2,1], alpha=1, iterations=1, batch_size=0)

        # three layers implicate 2 weight matrices
        numWeightMatrices = len(ex.weights)

        self.assertEqual(numWeightMatrices, 2)

    def testNetworkWith4Layers(self):
        ex = Network(sizes=[5,3,2,5], alpha=1, iterations=1, batch_size=0)

        # four layers implicate 3 weight matrices
        numWeightMatrices = len(ex.weights)

        self.assertEqual(numWeightMatrices, 3)

    def testNetworkWeightMatrixShape(self):
        ex = Network(sizes=[5,3,2,5], alpha=1, iterations=1, batch_size=0)

        # the weight matrix between layer 3 and 4 has to be of size (5, 2)
        weightMatrixShape = ex.weights[2].shape

        self.assertEqual(weightMatrixShape, (5, 2))

    def testOutputSizeOfAutoencoder(self):
        ex = Network(sizes=[5,3,2,5], alpha=1, iterations=1, batch_size=0)

        # the output should be of shape (5, 1)
        outputShape = ex.predict(np.array([[0.1, 0.1, 0.1, 0.1, 0.1]]).T).shape

        self.assertEqual(outputShape, (5, 1))

if __name__ == '__main__':
    unittest.main()