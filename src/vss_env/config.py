from dataclasses import dataclass


@dataclass
class SimConfig:
    # Rede do simulador
    sim_ip: str = "127.0.0.1"
    sim_port: int = 20011
    vision_ip: str = "224.0.0.1"
    vision_port: int = 10002
    field_type: str = "B"

    # Episódio
    max_steps: int = 600
    fps: int = 60

    # Obstáculos (robôs amarelos estáticos para o atacante desviar).
    # 0 = atacante sozinho no campo com a bola.
    num_obstacles: int = 0

    # Pesos das recompensas
    w_move: float = 0.2
    w_ball_grad: float = 0.2
    w_uvf: float = 0.6

    # Atuador
    max_speed: float = 1.5
    wheel_radius: float = 0.02
