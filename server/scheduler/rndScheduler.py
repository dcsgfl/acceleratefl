import random

class RNDSched:
    def __init__(self):
        pass

    def select_worker_instances(self, available_devices, client_threshold):
        selected_devices = {}
        device_keys = random.sample(list(available_devices.keys()), client_threshold)
        for key in device_keys:
            selected_devices[key] = available_devices[key].copy()
        return(selected_devices)
    
    class Factory:
        def get(self):
            return RNDSched()