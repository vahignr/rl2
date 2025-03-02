import os
import argparse
import time
from datetime import datetime
import gymnasium as gym
from gymnasium import error as gym_error
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Import torch if available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. Will use default PPO parameters.")

# Import from our other scripts
from record_videos import record_agent, record_random_agent, get_env_with_specific_resolution
from visualize_training import plot_training_results
from combine_videos import combine_videos, get_video_files

def setup_directories():
    """Create necessary directories"""
    dirs = ["logs", "logs/monitor", "models", "videos", "videos/random", "videos/checkpoints", "plots"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def create_env(monitor_dir, width=720, height=480):
    """Create a monitored environment with specified resolution"""
    def _init():
        env = get_env_with_specific_resolution("HumanoidStandup-v4", width, height)
        env = Monitor(env, monitor_dir)
        return env
    return _init

def train_agent(timesteps, record_interval=None, video_width=720, video_height=480, 
               final_video=True, final_video_path="training_video.mp4", make_video_every=None):
    """
    Train the agent and record videos periodically
    
    Args:
        timesteps: Total number of timesteps to train
        record_interval: Interval for recording videos during training
        video_width: Width of recorded videos
        video_height: Height of recorded videos
        final_video: Whether to create a combined video at the end
        final_video_path: Path to save the combined video
        make_video_every: Only make checkpoint videos every N intervals (to avoid too many videos)
    """
    # Setup directories
    setup_directories()
    
    # Current time for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create environment with monitoring
    monitor_dir = os.path.join("logs", "monitor")
    env = DummyVecEnv([create_env(monitor_dir, video_width, video_height)])
    
    # Set up the agent with better parameters for humanoid
    try:
        # Define base parameters that work without PyTorch
        ppo_params = {
            "policy": "MlpPolicy",
            "env": env,
            "verbose": 1,
            "tensorboard_log": "logs",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
        }
        
        # Add PyTorch-specific parameters if available
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
            
        # Create the model
        model = PPO(**ppo_params)
        
    except ImportError:
        # If tensorboard is not available, create the model without tensorboard logging
        print("TensorBoard not installed. Training will proceed without TensorBoard logging.")
        
        # Define base parameters without tensorboard
        ppo_params = {
            "policy": "MlpPolicy",
            "env": env,
            "verbose": 1,
            "tensorboard_log": None,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
        }
        
        # Add PyTorch-specific parameters if available
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
            
        # Create the model
        model = PPO(**ppo_params)
    
    # Setup callbacks for saving models
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000, record_interval // 2) if record_interval else 100_000,
        save_path="models",
        name_prefix=f"humanoid_standup_{timestamp}"
    )
    
    # Record videos of random agent for comparison
    print("\n--- Recording random agent performance ---")
    # Use width and height parameters instead of video_width and video_height
    record_random_agent(width=video_width, height=video_height)
    
    # Train the agent
    print(f"\n--- Starting training for {timesteps:,} timesteps ---")
    start_time = time.time()
    
    # Check if we can use progress bar
    use_progress_bar = True
    try:
        import tqdm
        import rich
    except ImportError:
        use_progress_bar = False
        print("Progress bar disabled - install tqdm and rich to enable it")
    
    if record_interval and record_interval > 0:
        # Train in intervals and record videos at each checkpoint
        intervals = max(1, timesteps // record_interval)
        interval_size = timesteps // intervals
        
        for i in range(intervals):
            print(f"\n--- Training interval {i+1}/{intervals} ---")
            model.learn(
                total_timesteps=interval_size, 
                callback=checkpoint_callback,
                progress_bar=use_progress_bar,
                reset_num_timesteps=False
            )
            
            # Save intermediate model
            intermediate_path = f"models/humanoid_standup_{timestamp}_step{(i+1)*interval_size}.zip"
            model.save(intermediate_path)
            
            # Only record video at specified intervals (to avoid too many videos for long training)
            should_record = True
            if make_video_every is not None:
                should_record = (i % make_video_every) == 0 or i == intervals - 1  # Always record last one
                
            if should_record:
                # Record video using the intermediate model
                print(f"\n--- Recording video at {(i+1)*interval_size:,} timesteps ---")
                checkpoint_video_dir = os.path.join("videos", "checkpoints", f"step_{(i+1)*interval_size}")
                
                # Get training statistics for overlay
                stats = {}
                if hasattr(model, "logger") and hasattr(model.logger, "name_to_value"):
                    for key in ["explained_variance", "learning_rate", "approx_kl", "clip_fraction"]:
                        if key in model.logger.name_to_value:
                            stats[key] = model.logger.name_to_value[key]
                
                # Use width and height parameters instead of video_width and video_height
                record_agent(
                    intermediate_path, 
                    video_folder=checkpoint_video_dir,
                    n_episodes=1,
                    width=video_width,
                    height=video_height,
                    custom_stats=stats if 'custom_stats' in record_agent.__code__.co_varnames else None
                )
    else:
        # Train without interruption
        model.learn(
            total_timesteps=timesteps, 
            callback=checkpoint_callback,
            progress_bar=use_progress_bar
        )
    
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n--- Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s ---")
    
    # Save the final model
    final_model_path = f"models/humanoid_standup_{timestamp}_final.zip"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Record video of the final model
    print("\n--- Recording final agent performance ---")
    # Use width and height parameters instead of video_width and video_height
    record_agent(
        final_model_path, 
        width=video_width, 
        height=video_height,
        n_episodes=5
    )
    
    # Plot training results
    print("\n--- Generating training plots ---")
    try:
        plot_training_results(monitor_dir)
    except Exception as e:
        print(f"Error generating training plots: {e}")
    
    # Create the combined training video if requested
    if final_video:
        print("\n--- Creating combined training video ---")
        try:
            # Get video files from different categories
            random_dir = os.path.join("videos", "random")
            checkpoints_dir = os.path.join("videos", "checkpoints")
            
            random_videos = get_video_files(random_dir) if os.path.exists(random_dir) else []
            checkpoint_videos = get_video_files(checkpoints_dir) if os.path.exists(checkpoints_dir) else []
            final_videos = [f for f in get_video_files("videos") 
                           if 'random' not in f and 'checkpoints' not in f]
            
            combine_videos(
                final_video_path,
                random_videos=random_videos,
                checkpoint_videos=checkpoint_videos,
                final_videos=final_videos,
                fps=30,
                add_titles=True,
                max_duration=15*60  # 15 minutes max
            )
            print(f"Combined video saved to {final_video_path}")
        except Exception as e:
            print(f"Error creating combined video: {e}")
    
    return final_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and record a Humanoid Standup agent')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Total number of timesteps to train (default: 1,000,000)')
    parser.add_argument('--record-interval', type=int, default=100_000, 
                        help='Interval for recording videos during training (default: 100,000, 0 to disable)')
    parser.add_argument('--video-width', type=int, default=720,
                        help='Width of recorded videos (default: 720)')
    parser.add_argument('--video-height', type=int, default=480,
                        help='Height of recorded videos (default: 480)')
    parser.add_argument('--no-final-video', action='store_true',
                        help='Disable creation of combined final video')
    parser.add_argument('--output-video', type=str, default="training_video.mp4",
                        help='Path to save the combined video (default: training_video.mp4)')
    parser.add_argument('--video-every', type=int, default=None,
                        help='Only record a video every N intervals (useful for long trainings)')
    
    args = parser.parse_args()
    
    final_model = train_agent(
        args.timesteps, 
        args.record_interval,
        args.video_width,
        args.video_height,
        not args.no_final_video,
        args.output_video,
        args.video_every
    )
    
    print("\nTraining and recording complete!")
    print(f"Final model: {final_model}")
    print("Videos saved in the 'videos' directory")
    if not args.no_final_video:
        print(f"Combined training video: {args.output_video}")
    print("Training plots saved in the 'plots' directory")