import numpy as np

class PYSched:
    def __init__(self):
        pass

    # Hellinger distance between two discrete distributions
    def hellingerDist(self, deviceP, deviceQ):
        sqrt2 = np.sqrt(2)
        sqrtdeviceP = np.sqrt(deviceP)
        sqrtdeviceQ = np.sqrt(deviceQ)
        sumOfSqrOfDiffSqrtdevice = np.sum((sqrtdeviceP - sqrtdeviceQ) ** 2)
        hellinger = np.sqrt(sumOfSqrOfDiffSqrtdevice) / sqrt2

        return hellinger

    # classification based histogram
    def histLabels(self, deviceLabel, nClasses):
        histres = np.histogram(deviceLabel, bins = np.arange(nClasses))
        return histres

    # Compare devices and return hellinger distance matrix
    #     deviceLabels - Label list from different device chunks
    #     nClasses - number of classes possible in the chunks
    def compareDevices(self, deviceLabels, nClasses):
        nDevices = len(deviceLabels)
        compMat = np.zeros([nDevices, nDevices])
        histInfo = [None]*nDevices

        deviceIds = deviceLabels.keys()
        histInfo = {}
        for id in deviceIds:
            histInfo[id] = self.histLabels(deviceLabels[id], nClasses)
            # print(histInfo[id])

        d = {}
        for i in deviceIds:
            idxi = 0
            d[i] = {}
            for j in deviceIds:
                if(j!=i):
                    idxj = 0
                    histdevi = histInfo[i]
                    histdevj = histInfo[j]
                    # print(type(histdevi[0]))
                    d[i][j] = self.hellingerDist(histdevi[0], histdevj[0])
                    # print(d[i])
            d[i] = dict(sorted(d[i].items(), key=lambda item: item[1]))
            #print(d[i])

        # for i in range(nDevices):
        #     histInfo[i] = histLabels(deviceLabels[i], nClasses)
        #
        # for i in range(nDevices):
        #     for j in range(nDevices):
        #         compMat[i][j] = hellingerDist(histInfo[i][0], histInfo[j][0])

        return(d)

    def select_worker_instances(self, _epoch, client_list, num_of_selected, dev_profile):    #p(y)
        if(_epoch==1):
            return(client_list)

        from mnistds import MNISTDS
        from realmnistds import REALMNISTDS
        cls = REALMNISTDS()

        deviceData = {}
        for dev in client_list:
            print(_epoch, dev.id)
            dev_data, dev_targets = cls.get_training_part(dev.id)
            ty=dev_targets.numpy()
            #print(dev.id,ty)
            deviceData[dev.id]=ty
        # print(deviceData)
        compMatrix = self.compareDevices(deviceData, 11)
        # print(compMatrix)
        selected_devid=set(compMatrix.keys())
        for kk in compMatrix.keys():
            if(len(compMatrix[kk])>0):
                ll=list(compMatrix[kk].keys())
                # print(kk, ll)
                dev1=dev_profile[ll[0]]
                dev0=dev_profile[kk]
                if(dev1<dev0):
                    selected_devid.remove(kk)
        res=[]
        for dev in client_list:
            if(dev.id in selected_devid):
                res.append(dev)
        return (res)
    
    class Factory:
        def get(self):
            return PYSched()