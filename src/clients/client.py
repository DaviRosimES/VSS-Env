import socket
from abc import ABC, abstractmethod

class Client(ABC):
    def __init__(self, server_address: str, server_port: int):
        self._client_socket : socket.socket = None
        self._server_address: str  = server_address
        self._server_port   : int  = server_port
        self._is_connected  : bool = False


    def connect(self):
        if not self._is_connected:
            self._connect_to_network()
            self._is_connected = True

    
    def close(self):
        if self._is_connected:
            self._disconnect_from_network()
            self._is_connected = False

    @abstractmethod
    def _connect_to_network(self):
        pass

    @abstractmethod
    def _disconnect_from_network(self):
        pass
