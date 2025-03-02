import os
import gymnasium as gym
from gymnasium import error as gym_error
from stable_baselines3 import PPO
import numpy as np
from gymnasium.wrappers import RecordVideo
import cv2
import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf  # For reading tensor values during recording

class VideoRecorderWithOverlay(RecordVideo):
    """Custom video recorder that adds text overlay to each frame."""
    
    def __init__(self, env, video_folder, episode_trigger=None, step_trigger=None, 
                 video_length=0, name_prefix="rl-video", metadata=None, 
                 model=None, show_stats=True):
        super().__init__(env, video_folder, episode_trigger, step_trigger, 
                          video_length, name_prefix, metadata)
        self.model = model
        self.episode_reward = 0
        self.episode_length = 0
        self.step_count = 0
        self.episode_num = 0
        self.show_stats = show_stats
        self.stats_data = {}
        
    def _add_text_to_frame(self, frame):
        """Add text overlay to the frame."""
        if not self.show_stats:
            return frame
            
        # Convert frame to PIL Image
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)
        
        # Try different fonts based on OS
        try:
            # Try to load a system font
            try:
                font = ImageFont.truetype("Arial", 14)
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans", 14)
                except:
                    # Fall back to default
                    font = ImageFont.load_default()
        except:
            # If all else fails, use default
            font = ImageFont.load_default()
        
        # Add episode and step info
        text_lines = [
            f"Episode: {self.episode_num}",
            f"Step: {self.step_count}",
            f"Reward: {self.episode_reward:.1f}",
            f"Length: {self.episode_length}"
        ]
        
        # Add training stats if model is provided
        if self.model and hasattr(self.model, "logger") and self.model.logger.name_to_value:
            # Get some key metrics if available
            for key in ["explained_variance", "learning_rate", "approx_kl", "clip_fraction"]:
                if key in self.stats_data:
                    value = self.stats_data[key]
                    text_lines.append(f"{key}: {value:.4f}")
        
        # Add timestamp
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text_lines.append(f"Time: {current_time}")
        
        # Draw text with semi-transparent background
        # Draw background rectangle for text
        x, y = 10, 10
        max_width = max([len(line) * 7 for line in text_lines])  # Approximate width
        rect_height = len(text_lines) * 18
        
        # Draw semi-transparent background
        overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([(x-5, y-5), (x + max_width + 5, y + rect_height + 5)], 
                              fill=(0, 0, 0, 128))
        pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_img)
        
        # Draw text
        for i, line in enumerate(text_lines):
            draw.text((x, y + i * 18), line, font=font, fill=(255, 255, 255))
        
        return np.array(pil_img)
    
    def step(self, action):
        """Override step to update stats and add overlay to frames"""
        obs, reward, terminated, truncated, info = super().step(action)
        
        self.episode_reward += reward
        self.episode_length += 1
        self.step_count += 1
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Override reset to update episode counter and reset rewards"""
        obs, info = super().reset(**kwargs)
        
        self.episode_num += 1
        self.episode_reward = 0
        self.episode_length = 0
        
        return obs, info
    
    def _video_enabled(self):
        """Override _video_enabled to add text overlay to frames"""
        is_enabled = super()._video_enabled()
        if is_enabled and hasattr(self, "env") and hasattr(self.env, "render_mode") and self.env.render_mode == "rgb_array":
            self.env.unwrapped._render_callback = self._add_text_to_frame
        return is_enabled
        
    def update_stats(self, stats_dict):
        """Update the statistics shown in the overlay"""
        self.stats_data = stats_dict

def get_env_with_specific_resolution(env_id, width=720, height=480, render_mode="rgb_array"):
    """Create a gymnasium environment with specific resolution for rendering"""
    try:
        # Try to use the latest version
        env = gym.make(env_id, render_mode=render_mode, width=width, height=height)
    except (gym_error.VersionNotFound, gym_error.UnregisteredEnv):
        # Fall back to v4 if v5 is not available
        env = gym.make(env_id, render_mode=render_mode, width=width, height=height)
    except TypeError:
        # If width and height are not supported, fall back to default
        try:
            env = gym.make(env_id, render_mode=render_mode)
            print(f"Warning: Could not set custom resolution for {env_id}. Using default resolution.")
        except (gym_error.VersionNotFound, gym_error.UnregisteredEnv):
            env = gym.make(env_id, render_mode=render_mode)
            print(f"Warning: Could not set custom resolution for {env_id}. Using default resolution.")
    
    return env

def record_agent(model_path, video_folder="videos", video_length=300, n_episodes=5, 
                width=720, height=480, show_stats=True, custom_stats=None):
    """
    Record videos of the agent's performance with detailed overlay.
    
    Args:
        model_path: Path to the trained model
        video_folder: Directory to save videos
        video_length: Max length of each episode in frames
        n_episodes: Number of episodes to record
        width: Width of the video (default: 720)
        height: Height of the video (default: 480)
        show_stats: Whether to show statistics overlay
        custom_stats: Dictionary of custom statistics to display
    """
    os.makedirs(video_folder, exist_ok=True)
    
    # Create environment with custom resolution
    env = get_env_with_specific_resolution("HumanoidStandup-v4", width, height)
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create video recorder with overlay
    video_env = VideoRecorderWithOverlay(
        env, 
        video_folder=video_folder,
        episode_trigger=lambda e: True,  # Record every episode
        name_prefix="humanoid_standup",
        model=model,
        show_stats=show_stats
    )
    
    if custom_stats:
        video_env.update_stats(custom_stats)
    
    for episode in range(n_episodes):
        obs, _ = video_env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (done or truncated) and steps < video_length:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = video_env.step(action)
            total_reward += reward
            steps += 1
            
            # Update overlay stats periodically
            if steps % 10 == 0 and hasattr(model, "logger") and model.logger.name_to_value:
                # Extract some training statistics for overlay
                stats = {}
                for key in ["explained_variance", "learning_rate", "approx_kl", "clip_fraction"]:
                    if key in model.logger.name_to_value:
                        stats[key] = model.logger.name_to_value[key]
                video_env.update_stats(stats)
            
        print(f"Episode {episode+1}: Total Reward: {total_reward}")
    
    video_env.close()
    print(f"Videos saved to {video_folder}")

def record_random_agent(video_folder="videos/random", video_length=300, n_episodes=2, 
                       width=720, height=480):
    """Record a random agent for comparison with custom resolution"""
    os.makedirs(video_folder, exist_ok=True)
    
    # Create environment with custom resolution
    env = get_env_with_specific_resolution("HumanoidStandup-v4", width, height)
    
    # Create video recorder with overlay
    video_env = VideoRecorderWithOverlay(
        env, 
        video_folder=video_folder,
        episode_trigger=lambda e: True,  # Record every episode
        name_prefix="random_agent",
        model=None,
        show_stats=True
    )
    
    for episode in range(n_episodes):
        obs, _ = video_env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (done or truncated) and steps < video_length:
            action = video_env.action_space.sample()
            obs, reward, done, truncated, info = video_env.step(action)
            total_reward += reward
            steps += 1
            
        print(f"Random Agent - Episode {episode+1}: Total Reward: {total_reward}")
    
    video_env.close()
    print(f"Random agent videos saved to {video_folder}")

if __name__ == "__main__":
    # Record random agent first
    record_random_agent(width=720, height=480)
    
    # Record trained agent using the final model if it exists
    model_path = "models/humanoid_standup_final.zip"
    if os.path.exists(model_path):
        record_agent(model_path, width=720, height=480)
    else:
        print(f"No model found at {model_path}. Train an agent first.")