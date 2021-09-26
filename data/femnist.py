import os
import sys
import json
import torch
import random

import numpy as np
import urllib.parse

from collections import defaultdict
from urllib.request import urlretrieve
from dataset import Dataset

######constants######
# Absolute path of "data" directory 
DATADIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

# Train and Test flag
TRAIN = 1
TEST = 0

class FEMNIST(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(DATADIR, 'femnist')
        self.xz = 'femnist.tar.xz'
        self.train_dir = os.path.join(self.path, 'train')
        self.test_dir = os.path.join(self.path, 'test')
                        
    class Factory:
        def get(self):
            return FEMNIST()
    
    def read_dir(self, data_dir):
        clients = []
        groups = []
        data = defaultdict(lambda: None)
        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            data.update(cdata['user_data'])
        clients = list(sorted(data.keys()))
        return clients, groups, data
    
    def read_data(self,):
        '''parses data in given train and test data directories
        assumes:
        - the data in the input directories are .json files with
            keys 'users' and 'user_data'
        - the set of train set users is the same as the set of test set users
        Return:
            clients: list of client ids
            groups: list of group ids; empty list if none found
            train_data: dictionary of train data
            test_data: dictionary of test data
        '''
        train_clients, train_groups, train_data = self.read_dir(self.train_dir)
        test_clients, test_groups, test_data = self.read_dir(self.test_dir)
        assert train_clients == test_clients
        assert train_groups == test_groups
        return train_clients, train_groups, train_data, test_data

    def download_data(self):

        # Untar femnist tarfile
        if 'train' not in os.listdir(self.path):
            res=os.popen('tar -xf '+ os.path.join(self.path, self.xz) +' -C '+ self.path).read()
            print(res)
        
        # get train and test data separately per user
        users, groups, train_data, test_data = self.read_data()

        #combine train and test data from all users
        self.train_x = []
        self.test_x = []
        self.train_y = []
        self.test_y = []
        for user in users:
            self.train_x = self.train_x + train_data[user]['x']
            self.train_y = self.train_y + train_data[user]['y']
            
            self.test_x = self.test_x + test_data[user]['x']
            self.test_y = self.test_y + test_data[user]['y']
        
        self.train_x = np.array([np.reshape(x, (28, 28).astype(float)) for x in self.train_x])
        self.test_x = np.array([np.reshape(x, (28, 28).astype(float)) for x in self.test_x])

        self.train_y = np.array(self.train_y).astype(int)
        self.test_y = np.array(self.test_y).astype(int)
         
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
    cls = FEMNIST()

    cls.download_data()
    cls.get_training_data(10)
