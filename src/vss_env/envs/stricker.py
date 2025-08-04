import math
import numpy as np
import gymnasium as gym
import random

from vss_env.clients.sim import ActuatorClient, VisionClient, ReplacerClient
from vss_env.entities import Field, Frame, Robot
from vss_env.proto.packet_pb2 import Packet
from vss_env.utils import Normalizer
from vss_env.uvf import UVF


class StrickerEnv(gym.Env):
    metadata = {"render_modes": ["None"], "render_fps": 0}

    def __init__(self):
        self.__frame: Frame = Frame()
        self.__field: Field = Field.from_type("B")
        self.__previous_ball_potential = None

        # Constantes de tempo
        self.__TIME_STEP = 1 / 60
        self.__current_step = 0
        self.__MAX_STEPS = 600  # 10s * 60fps

        # Pesos para as recompensas
        self.__W_MOVE = 0.2
        self.__W_BALL_GRAD = 0.2
        self.__W_UVF = 0.6

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(11,),
            dtype=np.float32
        )

        # Clients de comunicação com o FIRASim
        self.actuator_client = ActuatorClient("127.0.0.1", 20011, action_space=self.action_space)
        self.replacer_client = ReplacerClient("127.0.0.1", 20011, "B")
        self.vision_client = VisionClient("224.0.0.1", 10002, "B")

    def reset(self, seed=None, options=None):
        """
        Resete o episódio atual e gera um reposicionamento aleatório para os jogadores.
        :param seed: None
        :param options: None
        :return: Observation e info
        """
        self.__current_step = 0
        self.__previous_ball_potential = None

        # Envia posições aleatórias para o simulador
        replacer_packet = self.__create_replacement_packet()
        self.replacer_client.send_replacement(replacer_packet)

        # Aguarda alguns frames para garantir que a bola está bem posicionada
        self.__frame = self.vision_client.run_client()

        return self.__get_observation(), {}

    def step(self, actions):
        # Envia os comandos para todos os robôs
        commands = self.__convert_actions_to_commands(actions)
        self.actuator_client.send_commands(commands)

        # Aguarda o próximo frame do simulador
        self.__frame = self.vision_client.run_client()

        # Calcula a recompensa e verifica se o episódio terminou
        done = self._is_done()
        truncated = self._is_truncated()
        reward = self._calculate_reward()

        # Incrementa o contador de passos
        self.__current_step += 1

        return self.__get_observation(), reward, done, truncated, {}

    def close(self):
        """Fecha as conexões com os clientes para liberar recursos."""
        try:
            if self.actuator_client:
                self.actuator_client.close()
            if self.vision_client:
                self.vision_client.close()
            if self.replacer_client:
                self.replacer_client.close()
        except Exception as e:
            print(f"[ERROR] Erro ao fechar conexões: {e}")

    def render(self):
        # Não é necessário nenhuma implementação para renderizar já que o FIRASim sera nosso visualizador.
        pass

    def __convert_actions_to_commands(self, actions: dict) -> list:
        commands = []

        # Robô controlado (ID 2)
        v_left, v_right = self.actuator_client.actions_to_v_wheels(actions)
        commands.append(Robot(yellow_team=False, id=2, v_left_wheel=v_left, v_right_wheel=v_right))

        # Outros robôs
        for i in range(int(self.__field.NUM_ROBOTS / 2)):
            if i == 2:  # Pula o robô controlado
                continue
            v_left = 0.0
            v_right = 0.0
            team = False if i < int(self.__field.NUM_ROBOTS / 2) else True
            robot_id = i if i < int(self.__field.NUM_ROBOTS / 2) else i - int(self.__field.NUM_ROBOTS / 2)
            commands.append(Robot(yellow_team=team, id=robot_id, v_left_wheel=v_left, v_right_wheel=v_right))

        return commands

    def __create_replacement_packet(self):
        packet = Packet()
        packet.replace.ball.x, packet.replace.ball.y = self.replacer_client.random_ball_position()

        # Robôs azuis
        for i in range(int(self.__field.NUM_ROBOTS / 2)):
            robot_replacer = packet.replace.robots.add()
            robot_replacer.position.robot_id = i
            # Robo controlado (ID_2)
            if i == 2:
                robot_replacer.position.x, robot_replacer.position.y = self.replacer_client.random_robot_position()
            else:
                robot_replacer.position.x, robot_replacer.position.y = self.replacer_client.outside_robot_position()
            robot_replacer.position.orientation = random.uniform(0, 360)
            robot_replacer.yellowteam = False
            robot_replacer.turnon = True

        # Robôs amarelos
        for i in range(int(self.__field.NUM_ROBOTS / 2)):
            robot_replacer = packet.replace.robots.add()
            robot_replacer.position.robot_id = i
            robot_replacer.position.x, robot_replacer.position.y = self.replacer_client.outside_robot_position()
            robot_replacer.position.orientation = random.uniform(0, 360)
            robot_replacer.yellowteam = True
            robot_replacer.turnon = True

        return packet

    def __get_observation(self):
        ball_x = Normalizer.norm_pos_x(self.__frame.ball.x)
        ball_y = Normalizer.norm_pos_y(self.__frame.ball.y)
        ball_vx = Normalizer.norm_v(self.__frame.ball.v_x)
        ball_vy = Normalizer.norm_v(self.__frame.ball.v_y)
        robot = self.__frame.blue_robots.get(2)
        robot_x = Normalizer.norm_pos_x(robot.x)
        robot_y = Normalizer.norm_pos_y(robot.y)
        robot_vx = Normalizer.norm_v(robot.v_x)
        robot_vy = Normalizer.norm_v(robot.v_y)
        robot_orientation = robot.orientation
        robot_v_theta = Normalizer.norm_w(robot.v_orientation)

        return np.array([
            ball_x, ball_y,
            ball_vx, ball_vy,
            robot_x, robot_y,
            robot_vx,robot_vy,
            np.sin(robot_orientation),
            np.cos(robot_orientation),
            robot_v_theta
        ],dtype=np.float32)


    def _calculate_reward(self):
        # Recompensa/Penalidade por gol
        if self.__frame.ball.x > (self.__field.LENGTH / 2):
            reward = 100
        elif self.__frame.ball.x < -(self.__field.LENGTH / 2):
            reward = -100
        else:
            # Componentes existentes
            grad_ball_potential = self.__ball_grad()
            move_reward = self.__move_reward()
            uvf_reward = self.__uvf_reward()

            # Recompensa total
            reward = (
                    self.__W_MOVE * move_reward
                    + self.__W_BALL_GRAD * grad_ball_potential
                    + self.__W_UVF * uvf_reward
            )

        return reward

    def _is_done(self):
        """
        Verifica se aconteceu um gol no episódio.
        :return: True se o episódio terminou, False caso contrário.
        """
        if self.__frame.ball.x > (self.__field.LENGTH / 2) or self.__frame.ball.x < -(self.__field.LENGTH / 2):
            return True
        return False

    def _is_truncated(self):
        """
        Verifica se atingiu o tempo limite do episódio.
        """
        if self.__current_step >= self.__MAX_STEPS:
            return True

        return False

    def __uvf_reward(self) -> float:
        ball = np.array([self.__frame.ball.x, self.__frame.ball.y])
        uvf = UVF(field_width=self.__field.WIDTH, field_length=self.__field.LENGTH)
        opponents = self.__frame.yellow_robots

        robot = self.__frame.blue_robots.get(2)

        robot_pos = np.array([robot.x, robot.y])
        robot_vel = np.array([robot.v_x, robot.v_y])

        obstacles = []
        v_obstacles = []

        for opp in opponents.values():
            obstacles.append(np.array([opp.x, opp.y]))
            v_obstacles.append(np.array([opp.v_x, opp.v_y]))

        phi = uvf.get_phi(
            origin=robot_pos,
            target=ball,
            target_ori=0.0,
            v_robot=robot_vel,
            obstacles=obstacles,
            v_obstacles=v_obstacles
        )

        uvf_dir = np.array([np.cos(phi), np.sin(phi)])
        uvf_dir /= np.linalg.norm(uvf_dir)
        robot_dir = robot_vel / np.linalg.norm(robot_vel)

        alignment = np.dot(uvf_dir, robot_dir)

        return alignment

    def __ball_grad(self):
        """
        Calcula o gradiente do potencial da bola.
        """
        length_cm = self.__field.LENGTH * 100
        half_length = (self.__field.LENGTH / 2.0) + 0.1

        # Distância para a defesa
        dx_d = (half_length + self.__frame.ball.x) * 100
        # Distância para o ataque
        dx_a = (half_length - self.__frame.ball.x) * 100
        dy = self.__frame.ball.y * 100

        dist_1 = -math.sqrt(dx_a ** 2 + 2 * dy ** 2)
        dist_2 = math.sqrt(dx_d ** 2 + 2 * dy ** 2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        if self.__previous_ball_potential is not None:
            diff = ball_potential - self.__previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.__TIME_STEP, -5.0, 5.0)

        self.__previous_ball_potential = ball_potential
        return grad_ball_potential

    def __move_reward(self):
        """
        Calcula a recompensa pelo movimento em direção à bola.
        """
        ball = np.array([self.__frame.ball.x, self.__frame.ball.y])
        robot = np.array([self.__frame.blue_robots[2].x, self.__frame.blue_robots[2].y])  # Robô 2
        robot_vel = np.array([self.__frame.blue_robots[2].v_x, self.__frame.blue_robots[2].v_y])

        robot_ball = ball - robot
        robot_ball = robot_ball / np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)
        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward