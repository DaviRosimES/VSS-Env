from typing import Dict
from vss_env.entities.robot import Robot
from vss_env.entities.ball import Ball


class Frame:
    def __init__(self):
        self.ball: Ball = Ball()
        self.blue_robots: Dict[int, Robot] = {}
        self.yellow_robots: Dict[int, Robot] = {}

    def parse(self, sim_frame, n_blues=3, n_yellows=3):
        self.ball.x = sim_frame[0]
        self.ball.y = sim_frame[1]
        self.ball.z = sim_frame[2]
        self.ball.v_x = sim_frame[3]
        self.ball.v_y = sim_frame[4]
        self.ball.v_z = sim_frame[5]

        rbt_obs = 6

        for i in range(n_blues):
            robot = Robot()
            robot.id = i
            robot.x = sim_frame[6 + (rbt_obs * i) + 0]
            robot.y = sim_frame[6 + (rbt_obs * i) + 1]
            robot.v_x = sim_frame[6 + (rbt_obs * i) + 2]
            robot.v_y = sim_frame[6 + (rbt_obs * i) + 3]
            self.blue_robots[robot.id] = robot

        for i in range(n_yellows):
            robot = Robot()
            robot.id = i
            robot.x = sim_frame[6 + n_blues * rbt_obs + (rbt_obs * i) + 0]
            robot.y = sim_frame[6 + n_blues * rbt_obs + (rbt_obs * i) + 1]
            robot.v_x = sim_frame[6 + n_blues * rbt_obs + (rbt_obs * i) + 2]
            robot.v_y = sim_frame[6 + n_blues * rbt_obs + (rbt_obs * i) + 3]

            self.yellow_robots[robot.id] = robot