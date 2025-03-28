import socket
import threading

import numpy as np
from google.protobuf.message import DecodeError

from src.clients import Client
from src.entities import Frame, Robot, Ball, Field
from src.proto.packet_pb2 import Environment


class VisionClient(Client):
    def __init__(self, server_address: str, server_port: int, field_type : str):
        super().__init__(server_address, server_port)
        self.__environment_mutex = threading.Lock()
        self.__environment: Environment = Environment()
        self.__frame: Frame = Frame()
        self.__field: Field = Field.from_type(field_type)
        self.__observation: np.ndarray = np.zeros((11,), dtype=np.float32)

        # Constantes para normalização
        self.NORM_BOUND = 1.0
        self.MAX_SPEED = 2.0  # Velocidade linear máxima em m/s
        self.MAX_ANGULAR_SPEED = 100.0  # rad/s

        self.connect()

    def _connect_to_network(self):
        """Binds the socket to the defined network and joins a multicast group."""
        try:
            # Cria um socket UDP
            self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

            # Permitir reuso de endereço
            self._client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Faz o bind ao endereço e porta
            self._client_socket.bind((self._server_address, self._server_port))

            # Configura o grupo multicast
            multicast_group = socket.inet_aton(self._server_address)
            local_interface = socket.inet_aton("0.0.0.0")  # Interface padrão
            self._client_socket.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_ADD_MEMBERSHIP,
                multicast_group + local_interface
            )
            #print(f"[INFO] Visão conectada em {self._server_address}:{self._server_port}.")
        except Exception as e:
            print(f"[ERROR] Failed to connect to network: {e}")

    def _disconnect_from_network(self):
        """Leaves the multicast group and closes the socket."""
        if self._client_socket:
            try:
                # Remove o grupo multicast
                multicast_group = socket.inet_aton(self._server_address)
                local_interface = socket.inet_aton("0.0.0.0")
                self._client_socket.setsockopt(
                    socket.IPPROTO_IP,
                    socket.IP_DROP_MEMBERSHIP,
                    multicast_group + local_interface
                )
            except Exception as e:
                print(f"[ERROR] Error while leaving multicast group: {e}")
            finally:
                # Fecha o socket
                self._client_socket.close()
                self._client_socket = None
                print("[INFO] Visão desconectada.")

    def run_client(self):
        '''
        Recebe o pacote de visão e trata ele.
        :return: Frame e Observação
        '''
        try:
            # Recebe um único pacote
            data, sender = self._client_socket.recvfrom(2048)
            #print("[INFO] Pacote de visão recebido.")

            if not data:
                return None, None

            # Faz o parse do pacote
            environment = Environment()
            environment.ParseFromString(data)

            # Atualiza o ambiente
            with self.__environment_mutex:
                self.__environment = environment
                self.__fill_frame()
                self.__fill_obs_solo_agent()

            return self.get_frame(), self.get_observation()

        except DecodeError:
            print("[ERROR] Falha ao decodificar o pacote.")
            return None, None
        except Exception as e:
            print(f"[ERROR] Erro em run_client: {e}")
            return None, None

    def __fill_obs_solo_agent(self):
        """Preenche o array de observação normalizada apenas com os dados da bola e do robô azul número 2."""
        # Inicializa a lista de observação
        self.__observation = np.zeros((11,), dtype=np.float32)
        index = 0

        self.__observation[index] = self.__norm_pos(self.__frame.ball.x)
        index += 1
        self.__observation[index] = self.__norm_pos_y(self.__frame.ball.y)
        index += 1
        self.__observation[index] = self.__norm_v(self.__frame.ball.v_x)
        index += 1
        self.__observation[index] = self.__norm_v(self.__frame.ball.v_y)
        index += 1

        robot = self.__frame.blue_robots.get(2)
        self.__observation[index] = self.__norm_pos(robot.x)
        index += 1
        self.__observation[index] = self.__norm_pos_y(robot.y)
        index += 1
        self.__observation[index] = np.sin(robot.orientation)
        index += 1
        self.__observation[index] = np.cos(robot.orientation)
        index += 1
        self.__observation[index] = self.__norm_v(robot.v_x)
        index += 1
        self.__observation[index] = self.__norm_v(robot.v_y)
        index += 1
        self.__observation[index] = self.__norm_w(robot.v_orientation)
        index += 1


    def __fill_frame(self) -> None:
        self.__frame.ball = Ball(
            x=self.__environment.frame.ball.x,
            y=self.__environment.frame.ball.y,
            z=self.__environment.frame.ball.z,
            v_x=self.__environment.frame.ball.vx,
            v_y=self.__environment.frame.ball.vy,
            v_z=self.__environment.frame.ball.vz
        )

        # Preencher os robôs azuis
        for robot in self.__environment.frame.robots_blue:
            self.__frame.blue_robots[robot.robot_id] = Robot(
                id=robot.robot_id,
                yellow_team=False,
                x=robot.x,
                y=robot.y,
                orientation=robot.orientation,
                v_x=robot.vx,
                v_y=robot.vy,
                v_orientation=robot.vorientation
            )

        # Preencher os robôs amarelos
        for robot in self.__environment.frame.robots_yellow:
            self.__frame.yellow_robots[robot.robot_id] = Robot(
                id=robot.robot_id,
                yellow_team=True,
                x=robot.x,
                y=robot.y,
                orientation=robot.orientation,
                v_x=robot.vx,
                v_y=robot.vy,
                v_orientation=robot.vorientation
            )

    def get_frame(self) -> Frame:
        return self.__frame

    def __fill_observation(self):
        """Preenche o array de observação normalizada."""
        # Reinicia o array de observação
        self.__observation = np.zeros((46,), dtype=np.float32)

        index = 0  # Índice para controlar a posição no array

        # Informações da bola (4 elementos)
        self.__observation[index] = self.__norm_pos(self.__frame.ball.x)
        index += 1
        self.__observation[index] = self.__norm_pos_y(self.__frame.ball.y)
        index += 1
        self.__observation[index] = self.__norm_v(self.__frame.ball.v_x)
        index += 1
        self.__observation[index] = self.__norm_v(self.__frame.ball.v_y)
        index += 1

        # Robôs azuis (7 elementos por robô)
        for robot in self.__frame.blue_robots.values():
            self.__observation[index] = self.__norm_pos(robot.x)
            index += 1
            self.__observation[index] = self.__norm_pos_y(robot.y)
            index += 1
            self.__observation[index] = np.sin(robot.orientation)
            index += 1
            self.__observation[index] = np.cos(robot.orientation)
            index += 1
            self.__observation[index] = self.__norm_v(robot.v_x)
            index += 1
            self.__observation[index] = self.__norm_v(robot.v_y)
            index += 1
            self.__observation[index] = self.__norm_w(robot.v_orientation)
            index += 1

        # Robôs amarelos (7 elementos por robô)
        for robot in self.__frame.yellow_robots.values():
            self.__observation[index] = self.__norm_pos(robot.x)
            index += 1
            self.__observation[index] = self.__norm_pos_y(robot.y)
            index += 1
            self.__observation[index] = np.sin(robot.orientation)
            index += 1
            self.__observation[index] = np.cos(robot.orientation)
            index += 1
            self.__observation[index] = self.__norm_v(robot.v_x)
            index += 1
            self.__observation[index] = self.__norm_v(robot.v_y)
            index += 1
            self.__observation[index] = self.__norm_w(robot.v_orientation)
            index += 1

    def get_observation(self) -> np.ndarray:
        return self.__observation

    def __norm_pos(self, pos):
        return np.clip((pos / (self.__field.LENGTH / 2)), -self.NORM_BOUND, self.NORM_BOUND)

    def __norm_pos_y(self, pos_y):
        return np.clip((pos_y / (self.__field.WIDTH / 2)), -self.NORM_BOUND, self.NORM_BOUND)

    def __norm_v(self, v):
        return np.clip(v / self.MAX_SPEED, -self.NORM_BOUND, self.NORM_BOUND)

    def __norm_w(self, w):
        return np.clip(w / self.MAX_ANGULAR_SPEED, -self.NORM_BOUND, self.NORM_BOUND)
