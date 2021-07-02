# Dataset tests

import os
import sys
import unittest

# Add data folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data')
sys.path.append(pwd)

import DatasetFactory as dftry

class TestDataset:
    def test_cifar10_download(self):
        cifar10 = dftry.getDataset('CIFAR10')
        self.assertTrue(cifar10.download_data())
    
    def test_mnist_download(self):
        mnist = dftry.getDataset('MNIST')
        self.assertTrue(mnist.download_data())

if __name__ == 'main':
    unittest.main()