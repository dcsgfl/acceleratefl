import os
import gzip
import zipfile

import numpy as np

import urllib.parse

from urllib.request import urlretrieve
from dataset import Dataset
from random import seed
from random import random

######constants######
# Absolute path of "data" directory 
DATADIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

class MNIST_ROT(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(DATADIR, 'mnist_rot')
        self.url = 'http://www.iro.umontreal.ca/~lisa/icml2007data/'
        self.zip = ['mnist_rotation_new.zip']
        self.train_file = 'mnist_all_rotation_normalized_float_train_valid.amat'
        self.test_file = 'mnist_all_rotation_normalized_float_test.amat'
    
    class Factory:
        def get(self):
            return MNIST_ROT()
             
    def download_data(self):
        # Create mnist directory is non-existent
        os.makedirs(self.path, exist_ok=True)

        # Download mnist zip files if non-existent
        for zipf in self.zip:
            if zipf not in os.listdir(self.path):
                urlretrieve(urllib.parse.urljoin(self.url, zipf), os.path.join(self.path, zipf))
                print('Downloaded ', zipf, ' from ', self.url)
        
        # Retrieve train and test data from amat files
        with zipfile.ZipFile(os.path.join(self.path, zipf)) as zp:
            tmp = np.loadtxt(zp.open(self.train_file))
            self.train_x, self.train_y = tmp[:, :-1].copy().reshape(-1, 28, 28), tmp[:, -1].copy().astype(np.uint8)
            tmp = np.loadtxt(zp.open(self.test_file))
            print(self.train_x[0].shape)
            self.test_x, self.test_y = tmp[:, :-1].copy().reshape(-1, 28, 28), tmp[:, -1].copy().astype(np.uint8)
            self.train_y = self.train_y.reshape((-1, 1))
            self.test_y = self.test_y.reshape((-1, 1))

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
            self.indices_train[i] = np.where(np.isin(self.train_y, [i]))[0]
        for i in range(self.n_unique_labels):
            self.indices_test[i] = np.where(np.isin(self.test_y, [i]))[0]

        return True

    def get_training_data(self, id):
        return super().get_training_data(id)
    
    def get_testing_data(self, id):
        return super().get_testing_data(id)


if __name__ == '__main__':
    cls = MNIST_ROT()

    cls.download_data()
    cls.get_training_data(9)
