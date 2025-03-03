import math
import numpy as np
import gymnasium as gym
from collections import deque

from src.clients.sim import ActuatorClient, VisionClient
from src.clients.sim.replacer import ReplacerClient
from src.entities import Field, Frame


class StrickerEnv(gym.Env):
    metadata = {"render_modes": ["None"], "render_fps": 0}

    def __init__(self):
        self.frame: Frame = Frame()
        self.field: Field = Field.from_type("B")


        self.previous_ball_potential = None
        self.last_frame = None
        self.last_processed_frame = None  # Último frame processado
        self.sent_commands = None
        self.time_step = 1 / 60

        self.max_steps = 3600
        self.current_step = 0
        self.w_move = 0.2  # Peso para o movimento em direção à bola
        self.w_ball_grad = 0.8  # Peso para o gradiente do potencial da bola
        self.w_energy = 2e-4  # Peso para a penalidade de energia
        self.ball_positions = deque(maxlen=300)  # Armazena os ultimos 300 frames/5 segundos das posições da bola
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-1.2,
            high=1.2,
            shape=(40,),
            dtype=np.float32
        )
        self.actuator_client = ActuatorClient("127.0.0.1", 20011, action_space=self.action_space)
        self.replacer_client = ReplacerClient("127.0.0.1", 20011, "B")
        self.vision_client = VisionClient("224.0.0.1", 10002, "B")

    def reset(self, seed=None, options=None):
        # Reinicia o ambiente e retorna a observação inicial
        self.current_step = 0
        self.previous_ball_potential = None
        self.last_frame = None
        self.sent_commands = None

        # Envia posições aleatórias para o simulador
        self.replacer_client.send_replacement()

        # Aguarda o próximo frame do simulador
        while True:
            frame, observation = self.vision_client.run_client()
            if frame is not None:  # Verifica se o frame é válido
                self.last_processed_frame = frame
                break

        return observation

    def step(self, action):
        # Envia os comandos para todos os robôs
        self.actuator_client.send_actions(action)
        self.sent_commands = action  # Armazena os comandos enviados

        # Aguarda o próximo frame do simulador
        frame, observation = self.vision_client.run_client()

        # Calcula a recompensa e verifica se o episódio terminou
        reward, goal = self._calculate_reward(frame)
        done = self._is_done(frame)

        # Incrementa o contador de passos
        self.current_step += 1

        return observation, reward, done, False, {"goal": goal}

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

    def _calculate_reward(self, frame: Frame):
        """
        Calcula a recompensa com base no estado atual do ambiente.

        :param frame: Objeto Frame contendo as informações do ambiente.
        :return: Recompensa calculada.
        """
        reward = 0
        goal = False

        # Verifica se ocorreu um gol
        if frame.ball.x > (self.field.WIDTH / 2):
            reward = 10  # Recompensa por marcar um gol
            goal = True
        elif frame.ball.x < -(self.field.WIDTH / 2):
            reward = -10  # Penalidade por sofrer um gol
            goal = True
        else:
            if self.last_frame is not None:
                # Calcula o gradiente do potencial da bola
                grad_ball_potential = self.__ball_grad(frame)
                # Calcula a recompensa pelo movimento em direção à bola
                move_reward = self.__move_reward(frame)
                # Calcula a penalidade de energia
                energy_penalty = self.__energy_penalty()

                # Recompensa total ponderada
                reward = (
                        self.w_move * move_reward
                        + self.w_ball_grad * grad_ball_potential
                        + self.w_energy * energy_penalty
                )

        self.last_frame = frame
        return reward, goal

    def _is_done(self, frame: Frame):
        """
        Verifica se o episódio terminou.

        :param frame: Objeto Frame contendo as informações do ambiente.
        :return: True se o episódio terminou, False caso contrário.
        """
        # Verifica se ocorreu um gol
        if frame.ball.x > (self.field.WIDTH / 2) or frame.ball.x < -(self.field.WIDTH / 2):
            return True

        # Verifica se o número máximo de passos foi atingido
        if self.current_step >= self.max_steps:
            return True

        # Adiciona a posição atual da bola à fila
        self.ball_positions.append((frame.ball.x, frame.ball.y))

        # Verifica se todas as posições armazenadas são iguais
        if len(self.ball_positions) == self.ball_positions.maxlen and len(set(self.ball_positions)) == 1:
            return True  # Bola não se moveu nos últimos 300 frames

        return False

    def __ball_grad(self, frame: Frame):
        """
        Calcula o gradiente do potencial da bola.
        """
        length_cm = self.field.LENGTH * 100
        half_length = (self.field.LENGTH / 2.0) + 0.1

        # Distância para a defesa
        dx_d = (half_length + frame.ball.x) * 100
        # Distância para o ataque
        dx_a = (half_length - frame.ball.x) * 100
        dy = frame.ball.y * 100

        dist_1 = -math.sqrt(dx_a ** 2 + 2 * dy ** 2)
        dist_2 = math.sqrt(dx_d ** 2 + 2 * dy ** 2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        if self.previous_ball_potential is not None:
            diff = ball_potential - self.previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.time_step, -5.0, 5.0)

        self.previous_ball_potential = ball_potential
        return grad_ball_potential

    def __move_reward(self, frame: Frame):
        """
        Calcula a recompensa pelo movimento em direção à bola.
        """
        ball = np.array([frame.ball.x, frame.ball.y])
        robot = np.array([frame.blue_robots[2].x, frame.blue_robots[2].y])  # Robô 2
        robot_vel = np.array([frame.blue_robots[2].v_x, frame.blue_robots[2].v_y])

        robot_ball = ball - robot
        robot_ball = robot_ball / np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)
        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward

    def __energy_penalty(self):
        """
        Calcula a penalidade de energia com base nas velocidades das rodas.
        """
        if self.sent_commands is None:
            return 0

        en_penalty_1 = abs(self.sent_commands[0])
        en_penalty_2 = abs(self.sent_commands[1])
        energy_penalty = -(en_penalty_1 + en_penalty_2)
        return energy_penalty