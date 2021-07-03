# The model owner is considered as client
import os
import sys
import argparse
import threading

# Add common folder to path
pwd = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'common')
sys.path.append(pwd)

import devicetocentral_pb2_grpc
from utilities import Utility as util

class DeviceToCentralServicer(devicetocentral_pb2_grpc.DeviceToCentralServicer):
    
    def __init__(self):
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

