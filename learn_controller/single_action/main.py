import time

import gym
import torch


def main():
    from gym.envs.registration import register

    register(id="SimpleBallThrowing-v0", entry_point="env:SimpleBallThrowingEnv")

    env = gym.make("SimpleBallThrowing-v0", kwargs={"use_gui": True})
    for i in range(20):
        ob = env.reset()
        action = env.action_space.sample()
        ob, reward, done, _ = env.step(action)
        env.render()
        if done:
            print("#### DONE ####")
            # ob = env.reset()
            time.sleep(1 / 30)
        print("ob=", ob)
        print("reward=", reward)


if __name__ == "__main__":
    main()
