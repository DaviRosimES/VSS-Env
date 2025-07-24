from dataclasses import dataclass


@dataclass
class Robot:
    id : int = None
    yellow_team : bool = None
    x : float = None
    y : float = None
    orientation : float = None
    v_x : float = None
    v_y : float = None
    v_orientation : float = None
    v_left_wheel : float = None
    v_right_wheel : float = None