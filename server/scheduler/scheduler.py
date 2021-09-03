"""
scheduler.py

Parent scheduler class for flsys
UMN DCSG, 2021
"""

import math
from random import choices

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
    def notify_worker_update(self, all_devices):
        raise NotImplementedError

    """
    Called by the server each epoch when devices must be selected for training

    @param available_devices    The devices available for training this epoch
    @param client_threshold     Max number of devices allowed for training
    """
    def select_worker_instances(self, available_devices, client_threshold):
        raise NotImplementedError

    """
    --- NOT ABSTRACT ---
    Called by child implementations to perform scheduling common to the
    techniques that perform clustering.

    * Assumes there are cluster ids assigned to available_devices
    * Assumes self.cluster_ids contains the set of valid ids
    """
    def _schedule_clusters(self, clustInfo, available_devices, client_threshold):
        closs = {}
        cdevs = {}

        #
        # Compute the total loss in each cluster
        #
        for devId in available_devices:

            dev = available_devices[devId]
            clustId = dev["cluster"]
            sqloss = math.pow(dev['loss'], 2.0)

            if clustId not in closs.keys():
                closs[clustId] = sqloss
                cdevs[clustId] = []
            else:
                closs[clustId] += sqloss

            cdevs[clustId].append(dev.copy())

        #
        # Compute the average loss in each cluster and assign a
        # sampling probability based on the average loss.
        #
        clusters = []
        probs = []
        denom = 0.0
        for cid in closs.keys():
            closs[cid] /= clustInfo[cid]['count']
            denom += closs[cid]
            clusters.append(cid)
            probs.append(closs[cid])

        for idx in range(len(probs)):
            probs[idx] /= denom

        #
        # Sort the devices in each cluster by their utility
        #
        for cid in clusters:

            utility = []
            for dev in cdevs[cid]:
                #util = dev['loss'] / (1.0 + (dev['cpu_usage'] / 100.0))
                util = 1.0 - dev['cpu_usage']
                utility.append(util)

            s = [x for _, x in sorted(zip(utility, cdevs[cid]),
                                      key=lambda pair: pair[0],
                                      reverse=True)]
            cdevs[cid] = s

        #
        # Sample clusters and select the best devices
        #
        selected = {}

        #print(clusters)
        #print(probs)

        count = len(clusters)
        for i in range(count):

            idx = choices(range(len(clusters)), weights=probs)[0]
            clust = clusters[idx]
            devs = cdevs[clust]
            dev = devs[0]    # this will just work if sorted jsw
            selected[dev['id']] = dev

            del devs[0]
            if len(devs) == 0:

                # Remove this cluster from consideration
                del clusters[idx]
                p = probs[idx]
                del probs[idx]

                for i2 in range(len(probs)):
                    probs[i2] += (p / float(len(probs)))

        #print(selected.keys())

        return selected

