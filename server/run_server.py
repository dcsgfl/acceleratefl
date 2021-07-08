# The model owner is considered as client
import os
import sys
import argparse
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker

# Add common folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common')
sys.path.append(pwd)

import devicetocentral_pb2_grpc
from utilities import Utility as util

class DeviceToCentralServicer(devicetocentral_pb2_grpc.DeviceToCentralServicer):
    
    def __init__(self):
        super().__init__()
        self.available_devices = {}
        self.lock = threading.Lock()
        
    def RegisterToCentral(self, request, context):
        device_info = request
        msg = request.ip + ':' + request.flport
        id = util.get_id(msg)
        
        self.lock.acquire()
        if id not in self.available_devices:
            self.available_devices[id] = {}
            self.available_devices[id]['ip'] = request.ip
            self.available_devices[id]['flport'] = request.flport
        self.lock.release()

        return devicetocentral_pb2_grpc.RegStatus(
            success = True,
        )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def set_worker_conn(hook, devcentral, verbose):
    worker_instances = {}
    for devid in devcentral.available_devices:
        dev = devcentral.available_devices[devid]
        kwargs_websocket = {'host' : dev['ip'], 'hook' : hook, 'verbose' : verbose}
        clientWorker = WebsocketClientWorker(id = devid, port = dev['flport'], **kwargs_websocket)
        clientWorker.clear_objects_remote()
        worker_instances.append(clientWorker)
    
    return worker_instances

def train_and_eval(worker_instances, args):
    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    model = Net().to(device)

def parse_arguments(args = sys.argv[1:]):

    #Parse arguments
    parser = argparse.ArgumentParser(description='Run FL using websocker client workers')

    parser.add_argument(
        '--epochs',
        type = int,
        default = 200,
        help = 'Number of epochs for FL',
    )

    parser.add_argument(
        '--fedepoch',
        type = int,
        default = 10,
        help = 'number of epochs on remote worker before fedavg',
    )

    parser.add_argument(
        '--lr',
        type = float,
        default = 0.1,
        help = 'learning rate',
    )

    parser.add_argument(
        '--cuda',
        action = 'store_true',
        help = 'use cuda if available',
    )

    parser.add_argument(
        '--seed',
        type = int, 
        default = 1,
        help = 'seed for torch random number generator',
    )

    parser.add_argument(
        '--savemodel',
        action = 'store_true',
        help = 'if set, save model'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='start websocket client worker in verbose mode: --verbose'
    )

    args = parser.parse_args(args = args)
    return args

if __name__ == '__main__':

    args = parse_arguments()

    hook = sy.TorchHook(torch)
    
    # TO DO - Initialize devcentral
    worker_instances = set_worker_conn(hook, devcentral, args.verbose)

    # TO DO - thread for accepting devices
    
