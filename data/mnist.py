import os
import sys
import gzip
import shutil
import torch

import numpy as np

import urllib.parse

from urllib.request import urlretrieve
from dataset import Dataset
from random import seed
from random import random

import matplotlib.pyplot as plt

######constants######
# Absolute path of "data" directory 
DATADIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

class MNIST(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(DATADIR, 'mnist')
        self.url = 'http://yann.lecun.com/exdb/mnist/'
        self.gz = [ 'train-images-idx3-ubyte.gz',
                    't10k-images-idx3-ubyte.gz',
                    'train-labels-idx1-ubyte.gz',
                    't10k-labels-idx1-ubyte.gz']
    
    class Factory:
        def get(self):
            return MNIST()
             
    def download_data(self):
        # Create mnist directory is non-existent
        os.makedirs(self.path, exist_ok=True)

        # Download mnist gz files if non-existent
        for gz in self.gz:
            if gz not in os.listdir(self.path):
                urlretrieve(urllib.parse.urljoin(self.url, gz), os.path.join(self.path, gz))
                print('Downloaded ', gz, ' from ', self.url)
        
        # Retrieve train and test data from gz files
        for gz in self.gz:
            with gzip.open(os.path.join(self.path, gz), 'rb') as gzobj:
                if gz == 'train-images-idx3-ubyte.gz':
                    # skip first 16 bytes (magicno(4), image count(4), row count(4) = 28, column count(4) = 28)
                    self.train_x = np.frombuffer(gzobj.read(), np.uint8, offset=16).reshape(-1, 28, 28)
                
                if gz == 't10k-images-idx3-ubyte.gz':
                    # skip first 16 bytes (magicno(4), image count(4), row count(4) = 28, column count(4) = 28)
                    self.test_x = np.frombuffer(gzobj.read(), np.uint8, offset=16).reshape(-1, 28, 28)
                
                if gz =='train-labels-idx1-ubyte.gz':
                    # skip first 8 bytes (magicno(4), label count (4))
                    self.train_y = np.frombuffer(gzobj.read(), np.uint8, offset=8)

                if gz =='t10k-labels-idx1-ubyte.gz':
                    # skip first 8 bytes (magicno(4), label count (4))
                    self.test_y = np.frombuffer(gzobj.read(), np.uint8, offset=8)

        # remove the mnist directory after use
        # shutil.rmtree(self.path)
        fig = plt.figure
        plt.imshow(self.train_x[0], cmap='gray')
        plt.show()

        print(len(self.train_x))

        # get unique label count
        self.unique_labels = list(np.unique(self.train_y))
        self.n_unique_labels = len(self.unique_labels)
        self.min_label = min(self.unique_labels)
        self.max_label = max(self.unique_labels)

        # list of list: for both train and test
        #           inner list: indices corresponding to a specific label
        #           outer list: unique labels
        self.indices_train = [[] for x in range(self.n_unique_labels)]
        self.indices_test = [[] for x in range(self.n_unique_labels)]
        for i in range(self.n_unique_labels):
            self.indices_train[i] = np.isin(self.train_y, [i])
        for i in range(self.n_unique_labels):
            self.indices_test[i] = np.isin(self.test_y, [i])

        return True

    def get_training_data(self, id):
        return super().get_training_data(id)
    
    def get_testing_data(self, id):
        return super().get_testing_data(id)


if __name__ == '__main__':
    cls = MNIST()

    cls.download_data()
    cls.get_training_data(9)
