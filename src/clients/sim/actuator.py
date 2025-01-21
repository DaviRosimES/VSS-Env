import socket
from src.proto.packet_pb2 import Packet
from src.proto.command_pb2 import Command, Commands
from src.clients.client import Client



class ActuatorClient(Client):
    def __init__(self, server_address : str, server_port : int):
        super().__init__(server_address, server_port)
        self.connect()

    def _connect_to_network(self):
        self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._client_socket.connect((self._server_address, self._server_port))
            print(f"Connected to {self._server_address}:{self._server_port}")
        except socket.error as e:
            print(f"Failed to connect to {self._server_address}:{self._server_port} - {e}")

    def _disconnect_from_network(self):
        if self._client_socket:
            try:
                self._client_socket.close()
                print("Disconnected from network.")
            except socket.error as e:
                print(f"Error while disconnecting: {e}")
            finally:
                self._client_socket = None
    
    def send_command(self, robot_id, wheel_left, wheel_right):
        """Send command to the simulator."""
        # Check if the client is connected
        if not self._is_connected:
            self.connect()

        # Criando o pacote
        packet = Packet()
        command = packet.cmd.robot_commands.add()

        # Preenchendo os dados do comando
        command.id = robot_id
        command.yellowteam = True
        command.wheel_left = wheel_left
        command.wheel_right = wheel_right

        # Serializando o pacote
        buffer = packet.SerializeToString()

        # Enviando o pacote usando socket
        try:
            self._client_socket.sendall(buffer)
        except socket.error as e:
            print(f"[ERROR] Failed to send packet in Actuator: {e}")