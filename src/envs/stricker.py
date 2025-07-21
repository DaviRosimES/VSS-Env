import math
import numpy as np
import gymnasium as gym

from src.clients.sim import ActuatorClient, VisionClient
from src.clients.sim.replacer import ReplacerClient
from src.entities import Field, Frame
from src.utils.norm import Normalizer


class StrickerEnv(gym.Env):
    metadata = {"render_modes": ["None"], "render_fps": 0}

    def __init__(self):
        self.__frame: Frame = Frame()
        self.__field: Field = Field.from_type("B")

        self.__previous_ball_potential = None
        self.__sent_commands = None

        self.__TIME_STEP = 1 / 60
        self.__current_step = 0
        self.__MAX_STEPS = 600  # 10s * 60fps

        self.W_MOVE = 0.2  # Peso para o movimento em direção à bola
        self.W_BALL_GRAD = 0.8  # Peso para o gradiente do potencial da bola
        self.W_ENERGY = 2e-4  # Peso para a penalidade de energia

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
        self.__sent_commands = None

        # Envia posições aleatórias para o simulador
        self.replacer_client.send_replacement()

        # Aguarda alguns frames para garantir que a bola está bem posicionada
        self.__frame = self.vision_client.run_client()

        return self.__get_observation(), {}

    def step(self, action):
        # Envia os comandos para todos os robôs
        self.actuator_client.send_actions(action)
        self.__sent_commands = action  # Armazena os comandos enviados

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
            energy_penalty = self.__energy_penalty()

            # Recompensa total
            reward = (
                    self.W_MOVE * move_reward
                    + self.W_BALL_GRAD * grad_ball_potential
                    + self.W_ENERGY * energy_penalty
            )

        return reward

    def _is_done(self):
        """
        Verifica se aconteceu um gol no episódio.

        :return: True se o episódio terminou, False caso contrário.
        """
        # Verifica se ocorreu um gol
        if self.__frame.ball.x > (self.__field.LENGTH / 2) or self.__frame.ball.x < -(self.__field.LENGTH / 2):
            return True
        return False

    def _is_truncated(self):
        """
        Verifica se atingiu o tempo limite do episódio.
        :return:
        """
        # Verifica se o número máximo de passos foi atingido
        if self.__current_step >= self.__MAX_STEPS:
            return True

        return False

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

    def __energy_penalty(self):
        """
        Calcula a penalidade de energia com base nas velocidades das rodas.
        """
        if self.__sent_commands is None:
            return 0

        en_penalty_1 = abs(self.__sent_commands[0])
        en_penalty_2 = abs(self.__sent_commands[1])
        energy_penalty = -(en_penalty_1 + en_penalty_2)
        return energy_penalty