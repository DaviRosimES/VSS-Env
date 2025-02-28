import socket
import numpy as np

from src.proto.packet_pb2 import Packet
from src.entities import Robot
from src.noise import OrnsteinUhlenbeckAction
from src.clients import Client

class ActuatorClient(Client):
    def __init__(self, server_address: str, server_port: int, action_space, n_robots_blue: int = 3,
                 n_robots_yellow: int = 3):
        super().__init__(server_address, server_port)
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.MAX_VEL = 1.0  # Velocidade máxima normalizada
        self.WHEEL_RADIUS = 0.02  # Raio da roda (em metros)
        self.ou_actions = [
            OrnsteinUhlenbeckAction(action_space=action_space)
            for _ in range(n_robots_blue + n_robots_yellow)
        ]
        self.connect()

    def _connect_to_network(self):
        """Conecta ao simulador via UDP."""
        self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._client_socket.connect((self._server_address, self._server_port))
            print(f"[INFO] Actuator conectado em {self._server_address}:{self._server_port}.")
        except socket.error as e:
            print(f"[ERRO] Erro na conexão com o Actuator: {e}")

    def _disconnect_from_network(self):
        """Fecha a conexão com o simulador."""
        if self._client_socket:
            self._client_socket.close()
            print("[INFO] Actuator desconectado.")

    def send_actions(self, actions):
        """Envia os comandos para o simulador."""
        if not self._is_connected:
            self.connect()

        commands = self.__generate_commands(actions)
        packet = self.__create_packet(commands)
        self.__send_packet(packet)

    def __generate_commands(self, actions):
        """
        Gera comandos aletórios de velocidades para os robos não controlados, utiliza-se o processo de
        Ornstein-Uhlenbeck.
        Exclui o Robo ID_2, que é o robo no qual estamos controlando.
        """
        commands = []
        # Robô controlado (ID 2)
        v_left, v_right = self.__actions_to_v_wheels(actions)
        commands.append(Robot(yellow_team=False, id=2, v_left_wheel=v_left, v_right_wheel=v_right))

        # Outros robôs
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            if i == 2:  # Pula o robô controlado
                continue
            ou_action = self.ou_actions[i].sample()
            v_left, v_right = self.__actions_to_v_wheels(ou_action)
            team = False if i < self.n_robots_blue else True
            robot_id = i if i < self.n_robots_blue else i - self.n_robots_blue
            commands.append(Robot(yellow_team=team, id=robot_id, v_left_wheel=v_left, v_right_wheel=v_right))
        return commands

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

    def __actions_to_v_wheels(self, actions):
        """Converte ações em velocidades das rodas (rad/s)."""
        left = np.clip(actions[0], -self.MAX_VEL, self.MAX_VEL) / self.WHEEL_RADIUS
        right = np.clip(actions[1], -self.MAX_VEL, self.MAX_VEL) / self.WHEEL_RADIUS
        return left, right

    def __send_packet(self, packet):
        """Envia um pacote para o simulador."""
        try:
            self._client_socket.sendall(packet.SerializeToString())
            print("[INFO] Pacote enviado com sucesso!")
        except socket.error as e:
            print(f"[ERRO] Falha ao enviar pacote: {e}")