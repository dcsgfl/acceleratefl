import AlexNet
import DenseNet
import GoogleNet
import LeNet
import ResNet
import VGG
import WideResNet

class ModelFactory:
    factories = {}
    
    def addFactory(id, dftory):
        ModelFactory.factories.put[id] = dftory
    addFactory = staticmethod(addFactory)
    
    def getModel(id):
        if id not in ModelFactory.factories:
            ModelFactory.factories[id] = eval(id + '.Factory()')

        return ModelFactory.factories[id].get()
    getDataset = staticmethod(getModel)