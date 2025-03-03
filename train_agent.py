import os
import gymnasium as gym
from gymnasium import error as gym_error
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

def make_env():
    try:
        return gym.make("HumanoidStandup-v5")
    except (gym_error.VersionNotFound, gym_error.UnregisteredEnv):
        return gym.make("HumanoidStandup-v4")

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99
)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_dir, name_prefix="humanoid_standup_model")
TIMESTEPS = 1000000
use_progress_bar = True
try:
    import tqdm
    import rich
except ImportError:
    use_progress_bar = False
    print("[DEBUG] No progress bar installed")

model.learn(
    total_timesteps=TIMESTEPS,
    callback=checkpoint_callback,
    progress_bar=use_progress_bar
)
model.save(f"{models_dir}/humanoid_standup_final")
print("Training completed!")
