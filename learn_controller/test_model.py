import gym
import numpy as np
from const import base_dir
from gym.envs.registration import register
from stable_baselines3 import SAC, TD3

if __name__ == "__main__":
    register(id="SimpleBallThrowing-v0", entry_point="env:SimpleBallThrowingEnv")
    model_path = f"{base_dir}/Results/multi_action/final_model"
    model = SAC.load(model_path)
    env = gym.make("SimpleBallThrowing-v0", kwargs={"use_gui": True})
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if np.all(dones):
            break
