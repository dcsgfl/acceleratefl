import socket
from random import randint

class Utility:
    def get_free_port():
        while True:
            port = randint(32768, 61000)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if not (sock.connect_ex(('127.0.0.1', port)) == 0):
                return port