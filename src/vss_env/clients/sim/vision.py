import socket
import threading

from google.protobuf.message import DecodeError

from vss_env.clients import Client
from vss_env.entities import Frame, Robot, Ball, Field
from vss_env.proto.packet_pb2 import Environment


class VisionClient(Client):
    def __init__(self, server_address: str, server_port: int, field_type : str):
        super().__init__(server_address, server_port)
        self.__environment_mutex = threading.Lock()
        self.__environment: Environment = Environment()
        self.__frame: Frame = Frame()
        self.__field: Field = Field.from_type(field_type)

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
        """
        Recebe o pacote de visão e trata ele.
        :return: Frame e Observação
        """
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

            return self.get_frame()

        except DecodeError:
            print("[ERROR] Falha ao decodificar o pacote.")
            return None
        except Exception as e:
            print(f"[ERROR] Erro em run_client: {e}")
            return None


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