import torch.nn as nn
import torch.nn.functional as func


# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, (5,5), 1)
#         self.conv2 = nn.Conv2d(32, 64, (5,5), 1)
#         self.fc1 = nn.Linear(4 * 4 * 64, 2048)
#         self.fc2 = nn.Linear(2048, 62)

#     def forward(self, x):
#         x = func.relu(self.conv1(x))
#         x = func.max_pool2d(x, 2, 2)
#         x = func.relu(self.conv2(x))
#         x = func.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4 * 4 * 64)
#         x = func.relu(self.fc1(x))
#         x = self.fc2(x)
#         return func.log_softmax(x, dim=1)

#     class Factory:
#         def get(self):
#             return LeNet()

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()      # 1,  28x28       input [100,1,28,28]
        self.conv1=nn.Conv2d(1,10,5)        # 10, 24x24
        self.conv2=nn.Conv2d(10,20,3)       # 128, 10x10
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,62)

    def forward(self,x):
        in_size = x.size(0)
        x = func.max_pool2d(func.relu(self.conv1(x)), 2, 2)
        x = func.relu(self.conv2(x))
        x = x.view(in_size,-1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.fc2(x)
        return func.log_softmax(x,dim=1)

    class Factory:
         def get(self):
             return LeNet()