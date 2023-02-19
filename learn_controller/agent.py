import argparse
import csv
import json
import os
import random
import string
import time
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
import torch
from const import base_dir
from gym.envs.registration import register
from stable_baselines3 import PPO as PPO2
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecCheckNan,
    VecNormalize,
)


def random_string(length=10):
    """
    Generate a random string of given length
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


now = datetime.now()

# Specify save-directories
ENV_NAME = "PandaController"
# TIME_STAMP = now.strftime("_%Y_%d_%m__%H_%M_%S__%f")
TIME_STAMP = now.strftime("_%Y_%d_%m")
MODEL_ID = ENV_NAME + TIME_STAMP  # + random_string()
PATH = f"{base_dir}/Results/" + "multi_action" + "/"
SUFFIX = "final_model"
SAVE_MODEL_DESTINATION = PATH + SUFFIX  # For saving checkpoints and final model


# DEFAULTS:

RENDER = False
FIXED_NUM_REPETITIONS = True
CHECKPOINT_FREQUENCY = 10

parser = argparse.ArgumentParser()

parser.add_argument(
    "-r",
    "--retrain",
    "--restore",
    help="Path to zip archive for continuing training",
    type=str,
    default="",
)
parser.add_argument(
    "-p",
    "--params",
    "--parameters",
    help="Training parameter specifications",
    type=str,
    default=f"{base_dir}/agent_params.json",
)

args = parser.parse_args()

params = json.load(open(args.params))


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        print(f"Reducing lr to {progress_remaining * initial_value}")
        return progress_remaining * initial_value

    return func


def read_in_input_params(file_name):
    """
    Reads in parameter settings specified in file found under path_file_name. Then updates params specified above,
    accordingly.

    :param file_name: Path to file containing parameter specifications
    :return: -
    """

    with open(file_name) as json_file:
        data = json.load(json_file)
    params.update(data)


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        register(id=env_id, entry_point="env:SimpleBallThrowingEnv")
        env = Monitor(
            gym.make(
                env_id,
                # kwargs={'use_gui':True}
            ),
            filename=tb_log_dir,
        )
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    # Create and vectorize Environment

    seed = 20
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    np.seterr(all="raise")
    register(id="SimpleBallThrowing-v0", entry_point="env:SimpleBallThrowingEnv")

    tb_log_dir = f"{base_dir}/Results/tb_logs"
    use_single_env = True
    use_single_env = False

    if use_single_env:
        env = DummyVecEnv(
            [
                lambda: Monitor(
                    gym.make(
                        "SimpleBallThrowing-v0",
                        # kwargs={'use_gui':True}
                    ),
                    filename=tb_log_dir,
                )
            ]
        )  # The algorithms require a vectorized environment to run, hence vectorize
    else:
        num_cpu = 8  # Number of processes to use
        # # # # Create the vectorized environment
        env = SubprocVecEnv(
            [make_env("SimpleBallThrowing-v0", i) for i in range(num_cpu)]
        )
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)
    env = VecCheckNan(env, raise_exception=True)

    from stable_baselines3.common.noise import NormalActionNoise

    # Create the action noise object that will be used for exploration
    n_actions = env.action_space.shape[0]
    noise_std = 0.1
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
    )

    model = SAC(
        "MlpPolicy",
        env,
        buffer_size=params["buffer_size"],
        action_noise=action_noise,
        verbose=params["verbose"],
        device="cpu",
        learning_rate=linear_schedule(params["learning_rate"]),
        batch_size=params["batch_size"],
        train_freq=(params["train_freq"], "step"),
        tau=params["tau"],
        tensorboard_log=tb_log_dir,
        learning_starts=params["learning_starts"],
        policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=params["net_arch"]),
        seed=seed,
    )

    # Train the agent
    model.learn(
        total_timesteps=params["total_timesteps"],
        callback=None,
        progress_bar=True,
        log_interval=5,
    )

    from stable_baselines3.common.evaluation import evaluate_policy

    test_env = model.get_env()
    #  do not update them at test time
    test_env.training = False
    # reward normalization is not needed at test time
    test_env.norm_reward = False
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=3)
    print("mean_reward=", mean_reward)
    print("std_reward=", std_reward)

    model.save(SAVE_MODEL_DESTINATION)
