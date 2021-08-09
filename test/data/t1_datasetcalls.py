# Dataset tests

import os
import sys
import unittest

# Add data folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data')
sys.path.append(pwd)

from datasetFactory import DatasetFactory as dftry

class TestDataset(unittest.TestCase):
    # MNIST
    def test_mnist_download(self):
        self.skipTest("another method for skipping")
        print("mnist download test starting")
        mnist = dftry.getDataset('MNIST')
        self.assertTrue(mnist.download_data())
        print("mnist download test complete")
    
    def test_mnist_train(self):
        self.skipTest("another method for skipping")
        print("mnist train")
        mnist = dftry.getDataset('MNIST')
        self.assertTrue(mnist.download_data())
        tx, ty = mnist.get_training_data(10)
        if len(tx.size()) == 0 or len(ty.size()) == 0:
            self.assertTrue(False)
        else:
            self.assertTrue(True)
        print("mnist train complete")

    def test_mnist_test(self):
        self.skipTest("another method for skipping")
        print("mnist test")
        mnist = dftry.getDataset('MNIST')
        self.assertTrue(mnist.download_data())
        tx, ty = mnist.get_testing_data(10)
        if len(tx.size()) == 0 or len(ty.size()) == 0:
            self.assertTrue(False)
        else:
            self.assertTrue(True)
        print("mnist test complete")
    
    #CIFAR10
    def test_cifar10_download(self):
        self.skipTest("another method for skipping")
        print("cifar10 download test starting")
        cifar10 = dftry.getDataset('CIFAR10')
        self.assertTrue(cifar10.download_data())
        print("cifar10 download test complete")
    
    def test_cifar10_train(self):
        self.skipTest("another method for skipping")
        print("cifar10 train")
        cifar10 = dftry.getDataset('CIFAR10')
        self.assertTrue(cifar10.download_data())
        tx, ty = cifar10.get_training_data(10)
        if len(tx.size()) == 0 or len(ty.size()) == 0:
            self.assertTrue(False)
        else:
            self.assertTrue(True)
        print("cifar10 train complete")

    def test_cifar10_test(self):
        # self.skipTest("another method for skipping")
        print("cifar10 test")
        cifar10 = dftry.getDataset('CIFAR10')
        self.assertTrue(cifar10.download_data())
        tx, ty = cifar10.get_testing_data(10)
        if len(tx.size()) == 0 or len(ty.size()) == 0:
            self.assertTrue(False)
        else:
            self.assertTrue(True)
        print("cifar10 test complete")

if __name__ == '__main__':
    unittest.main()