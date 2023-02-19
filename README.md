# Robot control with differentiable simulation vs model-free RL

## Introduction

In this project, we compare the performance of model-free reinforcement learning (RL) and model-based RL using differentiable simulation. We use the [PyBullet framework](https://pybullet.org/) for the simulation and the [PyTorch framework](https://pytorch.org/) for the RL algorithms. Differentiable simulation is implemented using the [Tiny differentiable simulator](https://github.com/erwincoumans/tiny-differentiable-simulator).

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Get the necessary meshes and URDFs

### Validate installation

To test if RL loads properly, please use main.py files in:

1) single_action directory - single-joint control task.
2) the root directory - multi-joint control task.

## Run the simulation

Using the Makefile, you can run two control tasks in either training or evaluation mode.

### Task 1

Single joint control of the robot arm to throw a ball as far as possible. This task is meant for debugging purposes.

### Task 2

Double-joint control to throw a ball such that it hits a brick on the table.

## Experiments

Modifying contents of the agent_params.json and using the train script you could evaluate how different hyperparameters affect the performance of the robot. RL algorithms used in this project are PPO and SAC. Both appeared to be extremely sensitive to the choice of hyperparameters.

## Acknowledgements

- manipulating Franka Emika Panda robot - <https://github.com/hsp-iit/pybullet-robot-envs/tree/master/pybullet_robot_envs>
- linking Gym to Pybullet - <https://github.com/GerardMaggiolino/Gym-Medium-Post/blob/main/main.py>
- stable-baselines3 tutorials for fitting RL in Gym-like environment
- tiny-differentiable-simulator tutorials for using TDS with Pybullet - <https://github.com/erwincoumans/tiny-differentiable-simulator/tree/master/python/examples/whole_body_control>
