from stable_baselines3 import PPO
import gym
import gym_donkeycar
import time

from config import DONKEY_ENV_CONFIG


ENV_NAME = "donkey-generated-track-v0"


def main():
    env = gym.make(ENV_NAME, conf=DONKEY_ENV_CONFIG)
    model = PPO.load("../models/ppo_donkeycar")

    obs = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        time.sleep(0.03)

        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
