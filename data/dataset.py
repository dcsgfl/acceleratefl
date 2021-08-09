import sys
import torch
import random

import numpy as np

# Train and Test flag
TRAIN = 1
TEST = 0

class Dataset:
    def __init__(self) -> None:
        self.generated_dist_train = False
        self.generated_dist_test = False
        # raise NotImplementedError("ERROR: __init__ unimplemented")

    def download_data(self):
        raise NotImplementedError("ERROR: download_data unimplemented")
    
    def generate_data(self, id, flag):
        # associate a random constant label with current caller
        random.seed(1111 + int(id))
        my_label = random.randint(self.min_label, self.max_label)

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
            num_idxs = int(len(selected_noise_label_idxs) * p)
            pruned_selected_noise_label_idxs = selected_noise_label_idxs[:num_idxs]
            selected_noise_idxs.extend(pruned_selected_noise_label_idxs)

        # get index corresponding to my data label and take 90%
        all_my_label_idxs = tuple(np.where(scenario_index[my_label])[0])
        num_idxs = int(len(all_my_label_idxs) * 0.9)
        pruned_my_label_idxs = all_my_label_idxs[:num_idxs]

        # concatenate noise idx and my label index to generate final set of idx
        self.generated_data_idxs = np.concatenate([pruned_my_label_idxs , selected_noise_idxs])
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