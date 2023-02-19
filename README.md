# Robot control with differentiable simulation vs model-free RL

## Introduction

In this project, we compare the performance of model-free reinforcement learning (RL) and model-based RL using differentiable simulation. We use the [PyBullet framework](https://pybullet.org/) for the simulation and the [PyTorch framework](https://pytorch.org/) for the RL algorithms. Differentiable simulation is implemented using the [Tiny differentiable simulator](https://github.com/erwincoumans/tiny-differentiable-simulator).

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Get the necessary meshes and URDFs

## Run the simulation

Using the Makefile, you can run two control tasks in either training or evaluation mode.

### Task 1

Single joint control of the robot arm to throw a ball as far as possible. This task is meant for debugging purposes.

### Task 2

Double-joint control to throw a ball such that it hits a brick on the table.

## Experiments

Modifying contents of the agent_params.json and using the train script you could evaluate how different hyperparameters affect the performance of the robot. RL algorithms used in this project are PPO and SAC. Both appeared to be extremely sensitive to the choice of hyperparameters.
