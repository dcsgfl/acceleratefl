import random

class RNDSched:
    def __init__(self):
        pass

    def select_worker_instances(self, client_list, client_threshold):
        selected_client_list = random.sample(client_list, client_threshold)  # select num_of_selected dev for each round
        return(selected_client_list)
    
    class Factory:
        def get(self):
            return RNDSched()