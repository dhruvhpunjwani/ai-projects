import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gym
import gym_donkeycar

from config import DONKEY_ENV_CONFIG


ENV_NAME = "donkey-generated-track-v0"


def main():
    os.makedirs("../models", exist_ok=True)

    env = make_vec_env(
        ENV_NAME,
        n_envs=1,
        env_kwargs={"conf": DONKEY_ENV_CONFIG}
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="../logs/"
    )

    model.learn(total_timesteps=100000)
    model.save("../models/ppo_donkeycar")

    env.close()


if __name__ == "__main__":
    main()
