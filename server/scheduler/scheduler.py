"""
scheduler.py

Abstract scheduler class for flsys
UMN DCSG, 2021
"""


"""
All schedulers should implement the following interfaces
"""
class Scheduler:

    def __init__(self):
        pass

    """
    Called by the server when a worker is added or data distributions change

    @param all_devices  A list of all devices the server is aware of
    """
    def notify_worker_updates(self, all_devices):
        raise NotImplementedError

    """
    Called by the server each epoch when devices must be selected for training

    @param available_devices    The devices available for training this epoch
    @param client_threshold     Max number of devices allowed for training
    """
    def select_worker_instances(self, available_devices, client_threshold):
        raise NotImplementedError
