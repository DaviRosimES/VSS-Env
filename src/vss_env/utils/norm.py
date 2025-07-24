import numpy as np
from vss_env.entities import Field

class Normalizer:
    NORM_BOUND = 1.0
    MAX_SPEED = 1.5
    MAX_ANGULAR_SPEED = 75.0
    FIELD_LENGTH = Field.from_type("B").LENGTH
    FIELD_WIDTH = Field.from_type("B").WIDTH

    @staticmethod
    def norm_pos_x(pos_x):
        return np.clip((pos_x / (Normalizer.FIELD_LENGTH / 2)), -Normalizer.NORM_BOUND, Normalizer.NORM_BOUND)

    @staticmethod
    def norm_pos_y(pos_y):
        return np.clip((pos_y / (Normalizer.FIELD_WIDTH / 2)), -Normalizer.NORM_BOUND, Normalizer.NORM_BOUND)

    @staticmethod
    def norm_v(v):
        return np.clip(v / Normalizer.MAX_SPEED, -Normalizer.NORM_BOUND, Normalizer.NORM_BOUND)

    @staticmethod
    def norm_w(w):
        return np.clip(w / Normalizer.MAX_ANGULAR_SPEED, -Normalizer.NORM_BOUND, Normalizer.NORM_BOUND)