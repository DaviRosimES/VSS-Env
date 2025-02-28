import random
import math
import socket

from src.proto.packet_pb2 import Packet
from src.clients.client import Client
from src.entities.field import Field


class ReplacerClient(Client):
    def __init__(self, server_address: str, server_port: int, field_type : str):
        super().__init__(server_address, server_port)
        self.field : Field = Field.from_type(field_type)
        self.BALL_MARGIN = 0.4  # Margem para evitar bordas
        self.ROBOT_MARGIN = 0.1  # Margem para evitar bordas para os robôs
        self.connect()

    def _connect_to_network(self):
        self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._client_socket.connect((self._server_address, self._server_port))
            print(f"[INFO] Replacer conectado em {self._server_address}:{self._server_port}.")
        except socket.error as e:
            print(f"Erro na conexão: {e}")

    def _disconnect_from_network(self):
        self._client_socket.close()
        print("[INFO] Replacer desconectado.")

    def send_replacement(self):
        """Reposiciona a bola e os robôs em posições aleatórias no início do episódio."""
        if not self._is_connected:
            self.connect()

        packet = Packet()
        packet.replace.ball.x, packet.replace.ball.y = self.__random_ball_position()

        # Robôs azuis
        for i in range(3):
            robot_replacer = packet.replace.robots.add()
            robot_replacer.position.robot_id = i
            robot_replacer.position.x, robot_replacer.position.y = self.__random_robot_position()
            robot_replacer.position.orientation = random.uniform(0, 360)
            robot_replacer.yellowteam = False
            robot_replacer.turnon = True

        # Robôs amarelos
        for i in range(3):
            robot_replacer = packet.replace.robots.add()
            robot_replacer.position.robot_id = i
            robot_replacer.position.x, robot_replacer.position.y = self.__random_robot_position()
            robot_replacer.position.orientation = random.uniform(0, 360)
            robot_replacer.yellowteam = True
            robot_replacer.turnon = True

        buffer = packet.SerializeToString()
        try:
            self._client_socket.sendall(buffer)
            print("[INFO] Reposicionamento enviado!")
        except socket.error as e:
            print(f"[ERRO] Falha ao enviar: {e}")

    def __random_ball_position(self):
        """Gera uma posição aleatória válida dentro do campo para a bola."""
        x = random.uniform(-self.field.WIDTH / 2 + self.BALL_MARGIN, self.field.WIDTH / 2 - self.BALL_MARGIN)
        y = random.uniform(-self.field.LENGTH / 2 + self.BALL_MARGIN, self.field.LENGTH / 2 - self.BALL_MARGIN)
        return x, y

    def __random_robot_position(self):
        """Gera uma posição aleatória válida dentro do campo para os robôs."""
        x = random.uniform(-self.field.WIDTH / 2 + self.ROBOT_MARGIN, self.field.WIDTH / 2 - self.ROBOT_MARGIN)
        y = random.uniform(-self.field.LENGTH / 2 + self.ROBOT_MARGIN, self.field.LENGTH / 2 - self.ROBOT_MARGIN)
        return x, y