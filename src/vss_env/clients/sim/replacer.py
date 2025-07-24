import random
import socket

from vss_env.proto.packet_pb2 import Packet
from vss_env.clients import Client
from vss_env.entities import Field


class ReplacerClient(Client):
    def __init__(self, server_address: str, server_port: int, field_type : str):
        super().__init__(server_address, server_port)
        self.__field : Field = Field.from_type(field_type)
        self.__BALL_MARGIN = 0.02  # Margem para evitar bordas [metros]
        self.__ROBOT_MARGIN = 0.04  # Margem para evitar bordas para os robôs [metros]
        self.connect()

    def _connect_to_network(self):
        self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._client_socket.connect((self._server_address, self._server_port))
            #print(f"[INFO] Replacer conectado em {self._server_address}:{self._server_port}.")
        except socket.error as e:
            print(f"[ERRO] Erro na conexão: {e}")

    def _disconnect_from_network(self):
        self._client_socket.close()
        #print("[INFO] Replacer desconectado.")

    def send_replacement(self, packet : Packet):
        """Reposiciona a bola e os robôs em posições aleatórias no início do episódio."""
        try:
            self._client_socket.sendall(packet.SerializeToString())
            #print("[INFO] Reposicionamento enviado!")
        except socket.error as e:
            print(f"[ERRO] Falha ao enviar: {e}")

    def random_ball_position(self):
        """Gera uma posição aleatória válida dentro do campo para a bola."""
        x = random.uniform(-self.__field.LENGTH / 2 + self.__BALL_MARGIN, self.__field.LENGTH / 2 - self.__BALL_MARGIN)
        y = random.uniform(-self.__field.WIDTH / 2 + self.__BALL_MARGIN, self.__field.WIDTH / 2 - self.__BALL_MARGIN)
        return x, y

    def random_robot_position(self):
        """Gera uma posição aleatória válida dentro do campo para os robôs."""
        x = random.uniform(-self.__field.LENGTH / 2 + self.__ROBOT_MARGIN, self.__field.LENGTH / 2 - self.__ROBOT_MARGIN)
        y = random.uniform(-self.__field.WIDTH / 2 + self.__ROBOT_MARGIN, self.__field.WIDTH / 2 - self.__ROBOT_MARGIN)
        return x, y

    def outside_robot_position(self):
        x = 10
        y = 10
        return x, y