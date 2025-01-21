import socket
import threading
from google.protobuf.message import DecodeError
from src.clients.client import Client
from src.proto.packet_pb2 import Environment

class VisionClient(Client):
    def __init__(self, server_address : str, server_port : int):
        super().__init__(server_address, server_port)
        self._environment_mutex = threading.Lock()
        self._last_environment = None
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
            print(f"Socket bound to {self._server_address}:{self._server_port}")

            # Configura o grupo multicast
            multicast_group = socket.inet_aton(self._server_address)
            local_interface = socket.inet_aton("0.0.0.0")  # Interface padrão
            self._client_socket.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_ADD_MEMBERSHIP,
                multicast_group + local_interface
            )
            print("Joined multicast group.")
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
                print("Left multicast group.")
            except Exception as e:
                print(f"[ERROR] Error while leaving multicast group: {e}")
            finally:
                # Fecha o socket
                self._client_socket.close()
                self._client_socket = None
                print("Socket closed.")
        

    def run_client(self):
        """Receives datagrams from the network, parses them, and updates the environment."""
        try:
            while True:
                print("[INFO] Info received from VisionClient.")
                # Recebe datagrama com tamanho máximo de 2048 bytes
                data, sender = self._client_socket.recvfrom(2048)

                # Verifica se o datagrama é válido (em Python, verificamos se há dados)
                if not data:
                    continue

                # Cria o objeto protobuf
                environment = Environment()

                try:
                    # Faz o parse dos dados recebidos no protobuf
                    environment.ParseFromString(data)
                except DecodeError:
                    print("[ERROR] Failure to parse protobuf data from datagram.")
                    continue

                # Atualiza o último ambiente recebido
                with self._environment_mutex:
                    self._last_environment = environment

        except Exception as e:
            print(f"[ERROR] Error in run_client: {e}")

