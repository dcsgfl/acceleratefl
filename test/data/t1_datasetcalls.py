# Dataset tests

import os
import sys
import unittest

# Add data folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data')
sys.path.append(pwd)

from datasetFactory import DatasetFactory as dftry

class TestDataset(unittest.TestCase):
    def test_cifar10_download(self):
        print("cifar10 download test starting")
        cifar10 = dftry.getDataset('CIFAR10')
        self.assertTrue(cifar10.download_data())
        print("cifar10 download test complete")
    
    def test_mnist_download(self):
        print("mnist download test starting")
        mnist = dftry.getDataset('MNIST')
        self.assertTrue(mnist.download_data())
        print("mnist download test complete")

if __name__ == '__main__':
    unittest.main()