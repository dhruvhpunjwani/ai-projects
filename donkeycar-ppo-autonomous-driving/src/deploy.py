"""
Deployment placeholder for running a trained PPO policy on edge hardware
such as Jetson Nano connected to a DonkeyCar platform.

In a real deployment:
- camera frames are captured from onboard sensors
- frames are preprocessed to match simulator training format
- PPO model predicts steering/throttle
- commands are sent to motor controllers
"""

from stable_baselines3 import PPO


def load_model(model_path="../models/ppo_donkeycar"):
    model = PPO.load(model_path)
    return model


if __name__ == "__main__":
    model = load_model()
    print("[INFO] PPO model loaded for deployment.")
