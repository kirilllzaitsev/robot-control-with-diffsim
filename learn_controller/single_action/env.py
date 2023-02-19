import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(os.path.dirname(currentdir))

import math
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
from gym.spaces.box import Box
from panda_env import pandaEnv
from utils import get_obj_state, print_pb_obj_state, render

sim_timestep = 1.0 / 240


import logging

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)


class SimpleBallThrowingEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, kwargs={}):
        # head rotation angle
        self.action_space = Box(low=np.array([0]), high=np.array([3.8]))
        # x,y of 1) ball before throw 2) target (EXTRA for now, since target is currently throw as far
        # as possible)
        # ball state before throw
        # r=(0.49336, 0.0007457, 0.8647)
        # q=((0.0071283, -0.0020773, -0.000221, 0.999972409))
        self.boundaries = np.array(
            [
                [0, 10],
                [0, 10],
                [0, 10],
            ]
        )
        # self.observation_space = Box(
        #     low=np.array(
        #         [self.boundaries[0, 0], self.boundaries[1, 0], self.boundaries[2, 0]]
        #     ),
        #     high=np.array(
        #         [self.boundaries[0, 1], self.boundaries[1, 1], self.boundaries[2, 1]]
        #     ),
        # )
        self.observation_space = Box(
            low=np.array([-np.inf] * 3), high=np.array([np.inf] * 3), dtype=np.float64
        )
        self.np_random, _ = gym.utils.seeding.np_random()
        if kwargs.get("use_gui", False):
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        self.robot = pandaEnv(self.client, use_IK=1)
        logging.debug(f"self.robot={self.robot}")
        self.init_ball_pos = np.array([0.5, 0.0, 0.65])
        self.ball = None
        self.origin = np.array([0, 0, 0])
        # is it correct given that initial height could result in big norm?
        self.prev_dist = self.calc_dist()
        self.seed()
        self.reset()

    def check_off_bounds(self):
        ball_pos, _ = get_obj_state(self.ball)

        for dim in range(self.boundaries.shape[0]):
            if (self.boundaries[dim, 0] > ball_pos[0]) or (
                ball_pos[0] > self.boundaries[dim, 1]
            ):
                logging.debug(f"Off boundaries: {ball_pos}")
                return True
        return False

    def calc_dist(self):
        if self.ball is not None:
            ball_pos, _ = get_obj_state(self.ball)
        else:
            ball_pos = self.init_ball_pos
        return np.linalg.norm(ball_pos - self.origin) ** 2

    def step(self, action):
        # 6: perform throw
        # head_pos = 3.8
        head_pos = action
        logging.debug(f"head_pos={head_pos}")

        # logging.debug("Pos BEFORE throw")
        # logging.debug("#" * 10)
        # cubePos, cubeOrn = get_obj_state(self.ball)
        # logging.debug(f"cubePos={cubePos}")
        # # logging.debug(f"cubeOrn={cubeOrn}")
        # logging.debug("#" * 10)

        self.robot.throw(head_pos)
        for _ in range(150):
            p.stepSimulation()
            #     render(robot)
            time.sleep(sim_timestep)
        # logging.debug("Pos AFTER throw")
        # logging.debug("#" * 10)
        # cubePos, cubeOrn = get_obj_state(self.ball)
        # logging.debug(f"cubePos={cubePos}")
        # # logging.debug(f"cubeOrn={cubeOrn}")
        # logging.debug("#" * 10)

        # target_p = [1.0, 0.0, 0.8]
        # theta = np.arctan(target_p[1] / target_p[0])
        # cd = 0.7
        # ch = 0.04
        # r = [cd * np.sin(theta), cd * np.cos(theta), ch]
        # vel_norm is a target for future optimization
        # vel_norm = np.sqrt(
        #     (9.81 * (target_p[0] ** 2 + target_p[1] ** 2))
        #     / (r[2] - target_p[2] - np.sqrt(target_p[0] ** 2 + target_p[1] ** 2))
        # )
        # TODO: applying the calculations above to a ball

        ball_pos, _ = get_obj_state(self.ball)
        dist_to_goal = np.linalg.norm(ball_pos - self.init_ball_pos) ** 2
        reward = dist_to_goal
        # logging.info(f"raw reward={reward}")
        self.prev_dist = dist_to_goal

        self.done = True
        # if the ball is lower than table and was thrown forward
        if ball_pos[0] < self.init_ball_pos[0] and ball_pos[2] < self.init_ball_pos[2]:
            logging.debug(f"Ball fell backward. Penalizing")
            reward = -reward * 3
        if ball_pos[0] > self.init_ball_pos[0] and ball_pos[2] < self.init_ball_pos[2]:
            logging.debug("Ball fell beyond the table. Perfect!")
            reward = reward * 3

        ob = np.array(ball_pos, dtype=np.float32)
        logging.info(f"reward={reward}")
        logging.info(f"ob={ob}")

        # logging.debug(f"Tweaking reward to be head_pos!!!")
        # reward = np.asarray(head_pos)
        # logging.debug(f"reward={reward}")

        return ob, reward, self.done, dict()

    def reset(self):
        self.done = False
        # transform scene into the one before throw
        p.resetDebugVisualizerCamera(2.1, 90, -30, [0.0, -0.0, -0.0])
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)

        p.setTimeStep(sim_timestep)

        # Load plane contained in pybullet_data
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.ball = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "sphere_small.urdf"),
            self.init_ball_pos,
        )
        planeId = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "plane.urdf"), flags=flags
        )

        # Set gravity for simulation
        p.setGravity(0, 0, -9.8)

        self.robot = pandaEnv(self.client, use_IK=1)
        p.stepSimulation()

        p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.0]
        )

        for _ in range(100):
            p.stepSimulation()

        self.robot.pre_grasp()
        render(self.robot)
        p.stepSimulation()
        time.sleep(sim_timestep)

        # 1: go above the object
        pos_1 = [0.5, 0.0, 0.9]
        quat_1 = p.getQuaternionFromEuler([math.pi, 0, 0])

        self.robot.apply_action(pos_1 + list(quat_1))

        for _ in range(100):
            p.stepSimulation()
        #     render(self.robot)
        # time.sleep(sim_timestep)

        # 2: go down toward the object
        pos_2 = [0.5, 0.0, 0.67]
        quat_2 = p.getQuaternionFromEuler([math.pi, 0, 0])

        self.robot.apply_action(pos_2 + list(quat_2), max_vel=5)
        self.robot.pre_grasp()

        for _ in range(200):
            p.stepSimulation()
        #     render(self.robot)
        # time.sleep(sim_timestep)

        # 3: close fingers
        self.robot.grasp(self.ball)

        for _ in range(120):
            p.stepSimulation()
        #     render(self.robot)
        # time.sleep(sim_timestep)

        # 4: go up
        pos_4 = [0.5, 0.0, 0.9]
        quat_4 = p.getQuaternionFromEuler([math.pi, 0, 0])

        self.robot.apply_action(pos_4 + list(quat_4), max_vel=5)
        self.robot.grasp(self.ball)

        for _ in range(200):
            p.stepSimulation()
            #     render(self.robot)
            time.sleep(sim_timestep)

        self._observation = self.getExtendedObservation()

        return np.array(self._observation)

    def getExtendedObservation(self):
        """
        Construct an observation that serves as input to the reinforcement learning agent.
        :return: Representation of state of robotic arm and its surrounding
        """
        self._observation = []
        ball_pos, _ = get_obj_state(self.ball)
        self._observation.extend(ball_pos)

        return self._observation

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        p.disconnect()

    def __del__(self):
        ...
        # p.disconnect()


if __name__ == "__main__":
    import gym
    from gym.envs.registration import register

    register(id="SimpleBallThrowing-v0", entry_point="env:SimpleBallThrowingEnv")
    env = gym.make("SimpleBallThrowing-v0")
    env.action_space.sample()
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
