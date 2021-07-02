import argparse
import torch

import syft as sy
from syft.workers import websocket_server

def start_websocker_server_worker(id, host, port, dataset, hook, verbose):
    server = websocket_server.WebsocketServerWorker(
        id=id,
        host=host,
        port=port,
        hook=hook,
        verbose=verbose)

    
    



if __name__ == '__main__':

    #Parse arguments
    parser = argparse.ArgumentParser(description='Run websocket server worker')

    parser.add_argument(
        '--port',
        '-p',
        type=int,
        help='port number on which websocket server will listen: --port 8777',
    )

    parser.add_argument(
        '--host',
        '-h',
        default='localhost',
        help='host on which the websocket server worker should be run: --host 1.2.3.4',
    )

    parser.add_argument(
        '--id',
        type=str,
        help='name of the websocket server worker: --id alice'
    )

    parser.add_argument(
        '--dataset',
        '-ds',
        type=str,
        help='dataset used for the model: --dataset cifar10'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='start websocket server worker in verboase mode: --verbose'
    )

    args = parser.parse_args()

    # Hook PyTorch to add extra functionalities to support FL
    hook = sy.TorchHook(torch)

    server = start_websocker_server_worker(
        id = args.id,
        host = args.host,
        port = args.port,
        dataset = args.dataset,
        hook = hook,
        verbose = args.verbose
    )

