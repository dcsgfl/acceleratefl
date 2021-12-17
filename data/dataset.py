import sys
import torch
import random

import numpy as np

# Train and Test flag
TRAIN = 1
TEST = 0

class Dataset:

    def __init__(self) -> None:
        self.clear()

    def clear(self):
        self.generated_dist_train = False
        self.generated_dist_test = False
        self.generated_rot = 0.0

    def download_data(self):
        raise NotImplementedError("ERROR: download_data unimplemented")
    
    def generate_data(self, devid, maj_id, flag):

        # take 20 labels for 100 devices to get better clustering
        random.seed(int(devid))
        minlabel = self.min_label
        maxlabel = minlabel + 5

        # Ignoring the parm for now to avoid round robin.
        # This should add different sized clusters
        #my_label = maj_id
        my_label = random.randint(minlabel, maxlabel)
        #my_rot = random.choice([0.0, 15.0, 30.0])
        my_rot = random.choice([0.0, 30.0])

        # remove my lable from available ones for adding noise
        noise_labels = [*range(minlabel, maxlabel + 1, 1)]
        noise_labels.remove(my_label)

        # For maintaining same distribution across train and test, same noise percent should be added
        scenario_index = []
        rot = None
        if flag == TRAIN:
            scenario_index = self.indices_train
            rot = self.train_rot
        elif flag == TEST:
            scenario_index = self.indices_test
            rot = self.test_rot
        else:
            sys.exit("Incorrect flag for get_data")

        # get index corresponding to my data label and take 90%
        all_my_label_idxs = tuple(scenario_index[my_label])
        rot_my_label_idxs = []

        cnt = 0
        for idx, val in enumerate(all_my_label_idxs):
            if val and rot[idx] == my_rot:
                cnt = cnt + 1
                #rot_my_label_idxs.append(True)
                rot_my_label_idxs.append(idx)
            else:
                pass #rot_my_label_idxs.append(False)

        #print("ROT:",my_rot,"COUNT:",cnt)

        # take 80-90% data of my label and then add 12/7/6 % noise of the 80-90% data
        # 80-90% == 100 % train data
        percent = random.uniform(80.0, 90.0) / 100.0
        num_80_90_idxs = int(len(rot_my_label_idxs) * percent)

        #take 75% of 80-90% as majority
        num_75_idxs = int(num_80_90_idxs * 0.90)
        pruned_my_label_idxs = rot_my_label_idxs[:num_75_idxs]

        selected_noise_idxs = []
        noise_percents = [0.04, 0.03, 0.03]
        for p in noise_percents:
            # select a random noise label and remove it from existing noise list 
            selected_noise_label = random.choice(noise_labels)
            noise_labels.remove(selected_noise_label)
            selected_noise_label_idxs = tuple(np.where(scenario_index[selected_noise_label])[0])

            # extract only p% of selected noise label indices
            num_idxs = int(num_80_90_idxs * p)
            pruned_selected_noise_label_idxs = selected_noise_label_idxs[:num_idxs]
            selected_noise_idxs.extend(pruned_selected_noise_label_idxs)

        # concatenate noise idx and my label index to generate final set of idx
        #self.generated_data_idxs = np.concatenate([pruned_my_label_idxs , selected_noise_idxs])
        self.generated_data_idxs = np.array(pruned_my_label_idxs)
        np.random.shuffle(self.generated_data_idxs)

        self.generated_rot = my_rot

        if flag == TRAIN:
            self.generated_dist_train = True
            self.generated_train_idx = self.generated_data_idxs.astype(int)
        else:
            self.generated_dist_test = True
            self.generated_test_idx = self.generated_data_idxs.astype(int)
        

    def get_training_data(self, id):
        # check if data already generated
        minlabel = self.min_label
        maxlabel = minlabel + 10
        maj_id = int(id) % maxlabel
        if not self.generated_dist_train:
            self.generate_data(id, maj_id, TRAIN)
        idx = self.generated_train_idx

        # convert train data to tensor
        _tx = self.train_x[idx]
        _ty = self.train_y[idx]
        tx = torch.tensor(_tx).type('torch.FloatTensor')
        ty = torch.tensor(_ty, dtype=torch.int64)

        return(tx, ty, self.generated_rot)

    def get_testing_data(self, id):
        # check if data already generated
        minlabel = self.min_label
        maxlabel = minlabel + 10
        maj_id = int(id) % maxlabel
        if not self.generated_dist_test:
            self.generate_data(id, maj_id, TEST)
        idx = self.generated_test_idx
    
        # convert test data to tensor
        _tx = self.test_x[idx]
        _ty = self.test_y[idx]
        tx = torch.tensor(_tx).type('torch.FloatTensor')
        ty = torch.tensor(_ty, dtype=torch.int64)
        
        return(tx, ty, self.generated_rot)
