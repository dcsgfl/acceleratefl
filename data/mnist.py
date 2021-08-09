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

######constants######
# Absolute path of "data" directory 
DATADIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

# Train and Test flag
TRAIN = 1
TEST = 0

class MNIST(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(DATADIR, 'mnist')
        self.url = 'http://yann.lecun.com/exdb/mnist/'
        self.gz = [ 'train-images-idx3-ubyte.gz',
                    't10k-images-idx3-ubyte.gz',
                    'train-labels-idx1-ubyte.gz',
                    't10k-labels-idx1-ubyte.gz']
        
        self.generated_dist_train = False
        self.generated_dist_test = False
    
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
        shutil.rmtree(self.path)

        # get unique label count
        self.unique_labels = np.unique(self.train_y)
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

    def generate_data(self, id, flag):
        # associate a random constant label with current caller
        seed(1111 + int(id))
        my_label = random.randint(self.min_label, self.max_label+1)

        # remove my lable from available ones for adding noise
        noise_labels = self.unique_labels
        noise_labels.remove(my_label)

        # For maintaining same distribution across train and test, same noise percent should be added
        scenario_index = []
        if flag == TRAIN:
            scenario_index = self.indices_train
        elif flag == TEST:
            scenario_index = self.indices_test
        else:
            sys.exit("Incorrect flag for get_data")

        selected_noise_idxs = []
        noise_percents = [0.05, 0.03, 0.01]
        for p in noise_percents:
            # select a random noise label and remove it from existing noise list 
            selected_noise_label = random.choice(noise_labels)
            noise_labels.remove(selected_noise_label)

            selected_noise_label_idxs = tuple(np.where(scenario_index[selected_noise_label])[0])
            
            # extract only p% of selected noise label indices
            num_idxs = int(len(selected_noise_label_idxs[0]) * p)
            pruned_selected_noise_label_idxs = selected_noise_label_idxs[0][:num_idxs]
            selected_noise_idxs.extend(pruned_selected_noise_label_idxs)

        # get index corresponding to my data label and take 90%
        all_my_label_idxs = tuple(np.where(scenario_index[my_label])[0])
        num_idxs = int(len(all_my_label_idxs[0]) * 0.9)
        pruned_my_label_idxs = all_my_label_idxs[0][:num_idxs]

        # concatenate noise idx and my label index to generate final set of idx
        self.generated_data_idxs = np.concatenate(pruned_my_label_idxs)
        self.generated_data_idxs = np.concatenate([self.generated_data_idxs, selected_noise_idxs])
        np.random.shuffle(self.generated_data_idxs)

        if flag == TRAIN:
            self.generated_dist_train = True
            self.generated_train_idx = self.generated_data_idxs
        else:
            self.generated_dist_test = True
            self.generated_test_idx = self.generated_data_idxs
        

    def get_training_data(self, id):
        # check if data already generated
        if not self.generated_dist_train:
            self.generate_data(id, TRAIN)
        idx = self.generated_train_idx

        # convert train data to tensor
        _tx = self.train_x[idx]
        _ty = self.train_y[idx]
        tx = torch.tensor(_tx)
        ty = torch.tensor(_ty, dtype=torch.int64)

        return(tx, ty)

    def get_testing_data(self, id):
        # check if data already generated
        if not self.generated_dist_test:
            self.generate_data(id, TEST)
        idx = self.generated_test_idx

        # convert test data to tensor
        _tx = self.test_x[idx]
        _ty = self.test_y[idx]
        tx = torch.tensor(_tx)
        ty = torch.tensor(_ty, dtype=torch.int64)
        
        return(tx, ty)


if __name__ == '__main__':
    cls = MNIST()

    cls.download_data()
