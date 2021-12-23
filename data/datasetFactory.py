from cifar10 import CIFAR10
from mnist import MNIST
from femnist import FEMNIST
from mnist_rnd import MNIST_RND
from mnist_rot import MNIST_ROT
from mnist_rot_own import MNIST_ROT_OWN

class DatasetFactory:
    factories = {}
    
    def addFactory(id, dftory):
        DatasetFactory.factories.put[id] = dftory
    addFactory = staticmethod(addFactory)
    
    def getDataset(id):
        if id not in DatasetFactory.factories:
            DatasetFactory.factories[id] = eval(id + '.Factory()')

        return DatasetFactory.factories[id].get()
    getDataset = staticmethod(getDataset)
