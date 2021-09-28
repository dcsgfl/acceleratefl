import operator

from scheduler import Scheduler

class OORTSched(Scheduler):
    def __init__(self):
        pass

    def select_worker_instances(self, available_devices, client_threshold):
        selected_devices = {}
        device_keys = list(available_devices.keys())
        device_utils = dict((key, available_devices[key]['util']) for key  in device_keys)
        sorted_device_by_util= sorted(device_utils.items(), key=operator.itemgetter(1), reverse=True)
        n_selected_devices = 0
        for rec in sorted_device_by_util:
            n_selected_devices += 1
            selected_devices[rec[0]] = available_devices[rec[0]].copy()
            if n_selected_devices == client_threshold:
                break
        return(selected_devices)
    
    def notify_worker_update(self, all_devices):
        pass

    class Factory:
        def get(self):
            return OORTSched()
