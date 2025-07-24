import socket
import numpy as np

from vss_env.proto.packet_pb2 import Packet
from vss_env.clients import Client

class ActuatorClient(Client):
    def __init__(self, server_address: str, server_port: int, action_space, n_robots_blue: int = 3,
                 n_robots_yellow: int = 3):
        super().__init__(server_address, server_port)
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.MAX_SPEED = 1.5  # Velocidade linear máxima em m/s
        self.WHEEL_RADIUS = 0.02
        self.connect()

    def _connect_to_network(self):
        """Conecta ao simulador via UDP."""
        self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._client_socket.connect((self._server_address, self._server_port))
            #print(f"[INFO] Actuator conectado em {self._server_address}:{self._server_port}.")
        except socket.error as e:
            print(f"[ERRO] Erro na conexão com o Actuator: {e}")

    def _disconnect_from_network(self):
        """Fecha a conexão com o simulador."""
        if self._client_socket:
            self._client_socket.close()
            #print("[INFO] Actuator desconectado.")

    def send_commands(self, commands):
        """Envia os comandos para o simulador."""
        if not self._is_connected:
            self.connect()

        packet = self.__create_packet(commands)
        self.__send_packet(packet)

    def __create_packet(self, commands):
        """Cria um pacote a partir dos comandos recebidos."""
        packet = Packet()
        for cmd in commands:
            robot_cmd = packet.cmd.robot_commands.add()
            robot_cmd.id = cmd.id
            robot_cmd.yellowteam = cmd.yellow_team
            robot_cmd.wheel_left = cmd.v_left_wheel
            robot_cmd.wheel_right = cmd.v_right_wheel
        return packet

    def actions_to_v_wheels(self, actions):
        """Converte ações em velocidades das rodas (rad/s)."""
        left = np.clip(actions[0] * self.MAX_SPEED, -self.MAX_SPEED, self.MAX_SPEED) / self.WHEEL_RADIUS
        right = np.clip(actions[1] * self.MAX_SPEED, -self.MAX_SPEED, self.MAX_SPEED) / self.WHEEL_RADIUS
        return left, right

    def __send_packet(self, packet):
        """Envia um pacote para o simulador."""
        try:
            self._client_socket.sendall(packet.SerializeToString())
            #print("[INFO] Pacote enviado com sucesso!")
        except socket.error as e:
            print(f"[ERRO] Falha ao enviar pacote: {e}")