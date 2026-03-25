# Autonomous Driving with DonkeyCar and PPO

Reinforcement learning project using **Stable-Baselines3**, **PPO**, and the **DonkeyCar simulator** to train an autonomous driving agent for lane navigation and control.

## Project Overview

This project builds an RL-based autonomous driving pipeline that:

- trains a driving agent in the DonkeyCar simulator
- uses **Proximal Policy Optimization (PPO)** for policy learning
- evaluates learned driving behavior on generated tracks
- structures model deployment for edge devices such as **Jetson Nano**

## Technologies Used

- Python
- Stable-Baselines3
- PPO
- Gym
- gym-donkeycar
- PyTorch

## Project Structure

```text
donkeycar-ppo-autonomous-driving/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   └── deploy.py
└── examples/
    └── donkeycar_preview.jpg
```


## Installation
```text
pip install -r requirements.txt
```

## Training
```text
cd src
python train.py
```

## Evaluations
```text
cd src
python evaluate.py
```

## Notes
This project assumes access to the DonkeyCar simulator.
The deployment script is structured for edge-device integration such as Jetson Nano.
This repository is intended for educational and portfolio purposes.
