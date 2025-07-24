from dataclasses import dataclass

@dataclass
class Ball:
    x : float = None
    y : float = None
    z : float = None
    v_x : float = None
    v_y : float = None
    v_z : float = None