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
        # self.url = 'http://www-users.cselabs.umn.edu/~wang8662/'
        self.url = 'https://s3.amazonaws.com/nist-srd/SD19/by_class.zip'
        self.zip = 'femnist.zip'
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
        # Create cifar10 directory is non-existent
        os.makedirs(self.path, exist_ok=True)

        # Download femnist tar file is non-existent
        if self.zip not in os.listdir(self.path):
            urlretrieve(urllib.parse.urljoin(self.url, self.zip), os.path.join(self.path, self.zip))
            res=os.popen('unzip '+ os.path.join(self.path, self.zip) +' -d '+ self.path).read()
            print('Downloaded ', self.zip, ' from ', self.url)
        
        self.users, self.groups, self.train_data, self.test_data = self.read_data()

        return True

    # def generate_data(self, id, flag):
    #     # associate a random constant label with current caller
    #     # random.seed(1111 + int(id))
    #     # my_label = random.randint(self.min_label, self.max_label)

    #     # keep only 5 labels for 3 devices to get better clustering
    #     random.seed(int(id))
    #     minlabel = self.min_label
    #     maxlabel = minlabel + 4
    #     my_label = int(id)

    #     # remove my label from available ones for adding noise
    #     noise_labels = [*range(minlabel, maxlabel + 1, 1)]
    #     noise_labels.remove(my_label)

    #     # For maintaining same distribution across train and test, same noise percent should be added
    #     scenario_data = []
    #     if flag == TRAIN:
    #         scenario_data = self.train_data
    #     elif flag == TEST:
    #         scenario_data = self.test_data
    #     else:
    #         sys.exit("Incorrect flag for get_data")

    #     # get data corresponding to label my_label
    #     my_data_x = scenario_data[my_label]['x']
    #     n_data = len(my_data)
    #     n_prune_75 = int(n_data * 0.75)
    #     my_data_req = my_data[:n_prune_75]

    #     selected_noise_data = []
    #     noise_percents = [0.12, 0.07, 0.06]
    #     for p in noise_percents:
    #         # select a random noise label and remove it from existing noise list 
    #         selected_noise_label = random.choice(noise_labels)
    #         noise_labels.remove(selected_noise_label)
    #         selected_noise_full_data = scenario_data[selected_noise_label]['x']

    #         # extract only p% of selected noise label indices
    #         num_idxs = int(len(selected_noise_label_idxs) * p)
    #         pruned_selected_noise_label_idxs = selected_noise_label_idxs[:num_idxs]
    #         selected_noise_idxs.extend(pruned_selected_noise_label_idxs)

        

        # # concatenate noise idx and my label index to generate final set of idx
        # self.generated_data_idxs = np.concatenate([pruned_my_label_idxs , selected_noise_idxs])
        # np.random.shuffle(self.generated_data_idxs)

        # if flag == TRAIN:
        #     self.generated_dist_train = True
        #     self.generated_train_idx = self.generated_data_idxs
        # else:
        #     self.generated_dist_test = True
        #     self.generated_test_idx = self.generated_data_idxs

    def get_training_data(self, id):
        userid=self.users[id%5]
        _tx=self.train_data[userid]['x']
        
        _ty=self.train_data[userid]['y']
        num_of_samples = len(_ty)
        tx = torch.tensor(_tx).reshape(num_of_samples,28,28)
        ty = torch.tensor(_ty, dtype=torch.int64)
        
        return(tx, ty)
    
    def get_testing_data(self, id):
        userid=self.users[id]
        _tx=self.test_data[userid]['x']
        
        _ty=self.test_data[userid]['y']
        num_of_samples = len(_ty)
        tx = torch.tensor(_tx).reshape(num_of_samples,28,28)
        ty = torch.tensor(_ty, dtype=torch.int64)
        
        return(tx, ty)


if __name__ == '__main__':
    cls = FEMNIST()

    cls.download_data()
