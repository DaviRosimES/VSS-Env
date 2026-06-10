import math
import random

import gymnasium as gym
import numpy as np

from vss_env.clients.sim import ActuatorClient, ReplacerClient, VisionClient
from vss_env.config import SimConfig
from vss_env.entities import Field, Frame, Robot
from vss_env.proto.packet_pb2 import Packet
from vss_env.utils import Normalizer
from vss_env.uvf import UVF


class StrikerEnv(gym.Env):
    metadata = {"render_modes": ["None"], "render_fps": 0}

    # Quantidade de valores na observação por entidade.
    __BASE_OBS = 11  # bola (4) + atacante (7)
    __OBS_PER_OBSTACLE = 2  # posição (x, y) de cada obstáculo

    def __init__(self, config: SimConfig = None):
        if config is None:
            config = SimConfig()

        self.__frame: Frame = Frame()
        self.__field: Field = Field.from_type(config.field_type)
        self.__previous_ball_potential = None

        self.__TIME_STEP = 1 / config.fps
        self.__current_step = 0
        self.__MAX_STEPS = config.max_steps

        self.__W_MOVE = config.w_move
        self.__W_BALL_GRAD = config.w_ball_grad
        self.__W_UVF = config.w_uvf

        # Número de robôs amarelos colocados como obstáculos estáticos.
        # Limitado pela quantidade de robôs amarelos disponíveis no campo.
        max_obstacles = int(self.__field.NUM_ROBOTS / 2)
        self.__num_obstacles = max(0, min(config.num_obstacles, max_obstacles))
        # Distância mínima de spawn entre obstáculos, bola e atacante [metros].
        self.__OBSTACLE_MIN_DIST = 0.15

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        obs_size = self.__BASE_OBS + self.__OBS_PER_OBSTACLE * self.__num_obstacles
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(obs_size,), dtype=np.float32
        )

        self.actuator_client = ActuatorClient(
            config.sim_ip,
            config.sim_port,
            action_space=self.action_space,
            max_speed=config.max_speed,
            wheel_radius=config.wheel_radius,
        )
        self.replacer_client = ReplacerClient(
            config.sim_ip, config.sim_port, config.field_type
        )
        self.vision_client = VisionClient(
            config.vision_ip, config.vision_port, config.field_type
        )

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
        commands.append(
            Robot(yellow_team=False, id=2, v_left_wheel=v_left, v_right_wheel=v_right)
        )

        # Outros robôs
        for i in range(int(self.__field.NUM_ROBOTS / 2)):
            if i == 2:  # Pula o robô controlado
                continue
            v_left = 0.0
            v_right = 0.0
            team = False if i < int(self.__field.NUM_ROBOTS / 2) else True
            robot_id = (
                i
                if i < int(self.__field.NUM_ROBOTS / 2)
                else i - int(self.__field.NUM_ROBOTS / 2)
            )
            commands.append(
                Robot(
                    yellow_team=team,
                    id=robot_id,
                    v_left_wheel=v_left,
                    v_right_wheel=v_right,
                )
            )

        return commands

    def __create_replacement_packet(self):
        packet = Packet()

        # Posiciona a bola e o atacante (ID 2) sem sobreposição.
        ball_x, ball_y = self.replacer_client.random_ball_position()
        occupied = [(ball_x, ball_y)]
        striker_x, striker_y = self.__sample_free_position(occupied)
        occupied.append((striker_x, striker_y))

        # Amostra posições livres para os obstáculos amarelos estáticos.
        obstacle_positions = []
        for _ in range(self.__num_obstacles):
            pos = self.__sample_free_position(occupied)
            obstacle_positions.append(pos)
            occupied.append(pos)

        packet.replace.ball.x, packet.replace.ball.y = ball_x, ball_y

        num_team = int(self.__field.NUM_ROBOTS / 2)

        # Robôs azuis: apenas o atacante (ID 2) entra em campo.
        for i in range(num_team):
            robot_replacer = packet.replace.robots.add()
            robot_replacer.position.robot_id = i
            if i == 2:
                robot_replacer.position.x = striker_x
                robot_replacer.position.y = striker_y
            else:
                robot_replacer.position.x, robot_replacer.position.y = (
                    self.replacer_client.outside_robot_position()
                )
            robot_replacer.position.orientation = random.uniform(0, 360)
            robot_replacer.yellowteam = False
            robot_replacer.turnon = True

        # Robôs amarelos: os primeiros `num_obstacles` viram obstáculos em campo,
        # o restante é enviado para fora do campo.
        for i in range(num_team):
            robot_replacer = packet.replace.robots.add()
            robot_replacer.position.robot_id = i
            if i < self.__num_obstacles:
                robot_replacer.position.x, robot_replacer.position.y = (
                    obstacle_positions[i]
                )
            else:
                robot_replacer.position.x, robot_replacer.position.y = (
                    self.replacer_client.outside_robot_position()
                )
            robot_replacer.position.orientation = random.uniform(0, 360)
            robot_replacer.yellowteam = True
            robot_replacer.turnon = True

        return packet

    def __sample_free_position(self, occupied, max_tries=100):
        """Amostra uma posição de robô que respeite a distância mínima das já ocupadas."""
        x, y = self.replacer_client.random_robot_position()
        for _ in range(max_tries):
            if all(
                math.hypot(x - ox, y - oy) >= self.__OBSTACLE_MIN_DIST
                for ox, oy in occupied
            ):
                return x, y
            x, y = self.replacer_client.random_robot_position()
        # Fallback: aceita a última amostra mesmo sem o espaçamento ideal.
        return x, y

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

        observation = [
            ball_x,
            ball_y,
            ball_vx,
            ball_vy,
            robot_x,
            robot_y,
            robot_vx,
            robot_vy,
            np.sin(robot_orientation),
            np.cos(robot_orientation),
            robot_v_theta,
        ]
        observation.extend(self.__obstacles_observation(robot))

        return np.array(observation, dtype=np.float32)

    def __obstacles_observation(self, robot):
        """Posições (x, y) normalizadas dos obstáculos mais próximos do atacante.

        Retorna lista vazia quando o treino é sem obstáculos. Caso o frame reporte
        menos robôs do que o esperado, completa com 1.0 (sentinela "longe").
        """
        if self.__num_obstacles == 0:
            return []

        obstacles = sorted(
            self.__frame.yellow_robots.values(),
            key=lambda o: math.hypot(o.x - robot.x, o.y - robot.y),
        )[: self.__num_obstacles]

        values = []
        for obs in obstacles:
            values.append(Normalizer.norm_pos_x(obs.x))
            values.append(Normalizer.norm_pos_y(obs.y))

        # Padding defensivo para manter o tamanho fixo da observação.
        while len(values) < self.__OBS_PER_OBSTACLE * self.__num_obstacles:
            values.append(1.0)

        return values

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
        if self.__frame.ball.x > (self.__field.LENGTH / 2) or self.__frame.ball.x < -(
            self.__field.LENGTH / 2
        ):
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
            v_obstacles=v_obstacles,
        )

        robot_speed = np.linalg.norm(robot_vel)
        if robot_speed == 0:
            return 0.0

        uvf_dir = np.array([np.cos(phi), np.sin(phi)])
        robot_dir = robot_vel / robot_speed

        return float(np.dot(uvf_dir, robot_dir))

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

        dist_1 = -math.sqrt(dx_a**2 + 2 * dy**2)
        dist_2 = math.sqrt(dx_d**2 + 2 * dy**2)
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
        robot = np.array(
            [self.__frame.blue_robots[2].x, self.__frame.blue_robots[2].y]
        )  # Robô 2
        robot_vel = np.array(
            [self.__frame.blue_robots[2].v_x, self.__frame.blue_robots[2].v_y]
        )

        robot_ball = ball - robot
        robot_ball_norm = np.linalg.norm(robot_ball)
        if robot_ball_norm == 0:
            return 0.0
        robot_ball = robot_ball / robot_ball_norm

        move_reward = np.dot(robot_ball, robot_vel)
        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward
