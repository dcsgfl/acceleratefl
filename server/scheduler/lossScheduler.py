class LOSSched:
    def __init__(self):
        pass

    def stat_scheduler(_epoch, client_list, num_of_selected, dev_profile, evalloss):
        if(_epoch==1):
            return(client_list)

        sorted_evalloss={k: v for k, v in sorted(evalloss.items(), key=lambda item: item[1])}
        devids=[k for k in sorted_evalloss]
        devloss=[sorted_evalloss[k] for k in sorted_evalloss]
        devids=[devids[i:i+2] for i in range(0, len(devids), 2)]
        devloss=[devloss[i:i+2] for i in range(0, len(devloss), 2)]

        # print(devids)
        # print(devloss)

        selected_devid=[]
        for group in devids:
            if(dev_profile[group[0]]>dev_profile[group[1]]):
                selected_devid.append(group[1])
            else:
                selected_devid.append(group[0])

        res=[]
        for dev in client_list:
            if(dev.id in selected_devid):
                res.append(dev)
        return (res)

    class Factory:
        def get(self):
            return LOSSched()