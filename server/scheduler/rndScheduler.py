import random

class RNDSched:
    def __init__(self):
        pass

    def rand_scheduler(self, _epoch, client_list, num_of_selected):
        selected_client_list = random.sample(client_list, num_of_selected)  # select num_of_selected dev for each round
        return(selected_client_list)