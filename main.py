import os
import argparse
import time
from datetime import datetime
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# If torch is available, we do a bigger PPO net
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[DEBUG WARNING] PyTorch not found.")

from single_env_video_callback import SingleEnvVideoRecorderCallback
from visualize_training import plot_training_results
from combine_videos import combine_videos, get_video_files

def make_single_env(width=720, height=480):
    """
    Create a single Gym environment with render_mode='rgb_array', 
    so we can capture frames each step.
    """
    def _init():
        env_id = "HumanoidStandup-v4"
        env = gym.make(env_id, render_mode="rgb_array", width=width, height=height)
        # Wrap in Monitor so we get "info['episode']" at end of episodes
        env = Monitor(env)
        return env
    return _init

def train_single_env(timesteps, video_width, video_height, final_video_path, max_episodes=100):
    """
    Train a single environment from start to finish, capturing the entire run 
    in a single .mp4 via SingleEnvVideoRecorderCallback. 
    """
    os.makedirs("logs/monitor", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    env = DummyVecEnv([make_single_env(video_width, video_height)])

    # PPO hyperparams
    ppo_params = dict(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="logs",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )
    if TORCH_AVAILABLE:
        ppo_params.update({
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "policy_kwargs": dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.ReLU
            )
        })

    model = PPO(**ppo_params)

    # We'll do a big upper bound on timesteps, but rely on max_episodes to stop earlier if we want
    # e.g. if timesteps=50,000 but the environment only runs 20 episodes to get there, that's fine
    callback = SingleEnvVideoRecorderCallback(
        video_path=final_video_path,
        fps=30,
        max_episodes=max_episodes,
        verbose=1
    )

    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    training_time = time.time() - start_time
    h, rem = divmod(training_time, 3600)
    m, s = divmod(rem, 60)
    print(f"[DEBUG] Training completed in {int(h)}h {int(m)}m {s:.2f}s.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = f"models/humanoid_standup_single_env_{timestamp}.zip"
    model.save(final_model_path)
    print(f"[DEBUG] Final model saved to {final_model_path}")

    # Optionally we can also do a training plot, though we won't have multiple episodes in logs/monitor?
    try:
        plot_training_results("logs/monitor")
    except Exception as e:
        print(f"[DEBUG] Error generating training plots: {e}")

    return final_model_path

def main():
    parser = argparse.ArgumentParser("Single-Environment Full-Video Training")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Max total timesteps")
    parser.add_argument("--video-width", type=int, default=720, help="Video width")
    parser.add_argument("--video-height", type=int, default=480, help="Video height")
    parser.add_argument("--max-episodes", type=int, default=100, 
                        help="Stop after this many episodes, whichever first.")
    parser.add_argument("--output-video", type=str, default="training_single_env.mp4",
                        help="Output .mp4 filename for the entire run.")
    parser.add_argument("--no-final-video", action="store_true",
                        help="If set, skip combine_videos step (not that it's super relevant here).")

    args = parser.parse_args()

    final_model = train_single_env(
        timesteps=args.timesteps,
        video_width=args.video_width,
        video_height=args.video_height,
        final_video_path=args.output_video,
        max_episodes=args.max_episodes
    )

    print("\n[DEBUG] Single environment training done!")
    print(f"[DEBUG] Final model: {final_model}")
    print(f"[DEBUG] Single video: {args.output_video}")

    # If you STILL want to combine more videos (maybe you have a random agent video?), you can do so:
    if not args.no_final_video:
        print("\n[DEBUG] Combining videos (if you have any random or other videos in 'videos/') ...")
        random_dir = os.path.join("videos", "random")
        checkpoints_dir = os.path.join("videos", "checkpoints")
        random_videos = get_video_files(random_dir) if os.path.exists(random_dir) else []
        checkpoint_videos = get_video_files(checkpoints_dir) if os.path.exists(checkpoints_dir) else []
        final_videos = [f for f in get_video_files("videos") 
                        if 'random' not in f and 'checkpoints' not in f]
        # This might re-include your single env video or others
        combine_videos(
            "combined_single_env.mp4",
            random_videos=random_videos,
            checkpoint_videos=checkpoint_videos,
            final_videos=final_videos,
            fps=30,
            add_titles=True,
            max_duration=15*60
        )
        print("[DEBUG] combined_single_env.mp4 created.")
    else:
        print("[DEBUG] Skipping combine_videos step.")

if __name__ == "__main__":
    main()
