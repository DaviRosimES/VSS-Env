import numpy as np

PI = np.pi


def constrain_angle(x):
    x = (x + PI) % (2 * PI)
    if x <= 0:
        x += 2 * PI
    return x - PI


def get_gauss(r, delta):
    return np.exp(-(r ** 2) / (2 * (delta ** 2)))


class UVF:
    def __init__(self, field_width=1.3, field_length=1.5):
        self.field_width = field_width
        self.field_length = field_length

        self.k_kr = 0.25     # curvatura da espiral
        self.k_de = 0.05     # raio de transição da espiral
        self.k_delta = 0.1  # largura da transição gaussiana
        self.k_dmin = 0.05   # raio mínimo de ativação da repulsão
        self.k_o = 0.82      # peso da velocidade relativa

    def get_phih(self, p, tx, ty, ccw):
        signal = -1 if ccw else 1
        dx = p[0] - tx
        dy = p[1] - ty
        theta = np.arctan2(dy, dx)
        ro = np.hypot(dx, dy)

        if ro > self.k_de:
            phih = theta + signal * (PI / 2) * (2 - ((self.k_de + self.k_kr) / (ro + self.k_kr)))
        else:
            phih = theta + signal * (PI / 2) * np.sqrt(ro / self.k_de)
        return phih

    def add_field_walls(self, obstacles, v_obstacles):
        """Adiciona os limites do campo como obstáculos estáticos"""
        hw = self.field_width / 2
        hh = self.field_length / 2

        walls = [
            (0.0,  hh),  # superior
            (0.0, -hh),  # inferior
            (-hw, 0.0),  # esquerda
            ( hw, 0.0)   # direita
        ]
        obstacles += walls
        v_obstacles += [np.zeros(2) for _ in walls]
        return obstacles, v_obstacles

    def get_phi(self, origin, target, target_ori, obstacles, v_robot=np.array([0.0, 0.0]), v_obstacles=None):
        target_ori = constrain_angle(target_ori)
        rot = PI - target_ori

        def rotate(p):
            return np.dot([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]], p)

        # Adiciona os limites do campo como obstáculos
        if v_obstacles is None:
            v_obstacles = [np.zeros(2) for _ in obstacles]
        obstacles, v_obstacles = self.add_field_walls(obstacles, v_obstacles)

        origin_r = rotate(np.array(origin))
        target_r = rotate(np.array(target))

        x = origin_r[0] - target_r[0]
        y = origin_r[1] - target_r[1]

        y_l = y + self.k_de
        y_r = y - self.k_de

        p_l = (x, y_l)
        p_r = (x, y_r)

        if y < -self.k_de:
            phi_tuf = self.get_phih(p_l, 0, 0, ccw=True)
        elif y >= self.k_de:
            phi_tuf = self.get_phih(p_r, 0, 0, ccw=False)
        else:
            phi_hccw = self.get_phih(p_r, 0, 0, ccw=False)
            phi_hcw = self.get_phih(p_l, 0, 0, ccw=True)
            nh_cw = np.array([np.cos(phi_hcw), np.sin(phi_hcw)])
            nh_ccw = np.array([np.cos(phi_hccw), np.sin(phi_hccw)])
            phi_px = (abs(y_l) * nh_ccw[0] + abs(y_r) * nh_cw[0]) / (2.0 * self.k_de)
            phi_py = (abs(y_l) * nh_ccw[1] + abs(y_r) * nh_cw[1]) / (2.0 * self.k_de)
            phi_tuf = np.arctan2(phi_py, phi_px)

        for i, obstacle in enumerate(obstacles):
            obstacle = np.array(obstacle)
            v_obs = v_obstacles[i]

            s = self.k_o * (v_obs - v_robot)
            d_to_obs = np.linalg.norm(obstacle - origin)
            if np.linalg.norm(s) > d_to_obs:
                s = (obstacle - origin) / d_to_obs * d_to_obs

            virtual_obstacle = obstacle + s
            obs_r = rotate(virtual_obstacle)
            auf = origin_r - obs_r
            r = np.linalg.norm(auf)
            phi_auf = constrain_angle(np.arctan2(auf[1], auf[0]))
            phi_tuf = constrain_angle(phi_tuf)

            if r <= self.k_dmin:
                phi = phi_auf
            else:
                gauss = get_gauss(r - self.k_dmin, self.k_delta)
                diff = abs(phi_auf - phi_tuf)
                if diff > PI:
                    diff = abs(2 * PI - diff)
                cross = np.cross([np.cos(phi_auf), np.sin(phi_auf)], [np.cos(phi_tuf), np.sin(phi_tuf)])
                sgn = -1.0 if cross > 0 else 1.0
                phi = phi_tuf + sgn * diff * gauss

                vec_obs = obs_r - target_r
                if auf[0] < 0:
                    if not (
                            (vec_obs[1] > 0 > auf[1] and vec_obs[0] > 0) or
                            (vec_obs[1] < 0 < vec_obs[0] and auf[1] > 0) or
                            (vec_obs[1] > 0 and auf[1] >= 0) or
                            (vec_obs[1] < 0 and auf[1] <= 0)
                    ):
                        phi = phi_tuf + (phi_auf - phi_tuf) * gauss
            phi_tuf = phi

        phi = constrain_angle(phi_tuf - rot)
        return phi