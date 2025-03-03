import os
import cv2
import gymnasium as gym
from gymnasium import error as gym_error
import numpy as np
import datetime
from PIL import Image, ImageDraw, ImageFont

from stable_baselines3 import PPO

def get_env_with_specific_resolution(env_id, width=720, height=480, render_mode="rgb_array"):
    """
    Create a gymnasium environment that returns frames via env.render() with the desired resolution.
    """
    try:
        env = gym.make(env_id, render_mode=render_mode, width=width, height=height)
    except (gym_error.VersionNotFound, gym_error.UnregisteredEnv):
        env = gym.make(env_id, render_mode=render_mode)
    except TypeError:
        env = gym.make(env_id, render_mode=render_mode)
    return env

def add_text_overlay(
    frame, episode_num, step_count, episode_reward, episode_length,
    stats_data=None, show_stats=True
):
    """
    Add text overlay (Episode, Step, Reward, etc.) to the top-right corner of a frame (RGB).
    Returns a new annotated frame.
    """
    if frame is None or not show_stats:
        return frame

    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    frame_width, frame_height = pil_img.size

    # Attempt to load a TTF font; fallback if not found
    try:
        font = ImageFont.truetype("Arial", 14)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()

    # Build lines of text
    text_lines = [
        f"Episode: {episode_num}",
        f"Step: {step_count}",
        f"Reward: {episode_reward:.2f}",
        f"Length: {episode_length}",
    ]

    if stats_data:
        rename_map = {
            "explained_variance": "Expl Var",
            "learning_rate":      "LR",
            "approx_kl":         "KL Div",
            "clip_fraction":      "ClipFr",
        }
        for k, v in stats_data.items():
            label = rename_map.get(k, k)
            text_lines.append(f"{label}: {v:.4f}")

    # Timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text_lines.append(f"Time: {current_time}")

    # Compute bounding box for text in top-right corner
    line_widths = [draw.textlength(line, font=font) for line in text_lines]
    max_line_width = max(line_widths) if line_widths else 0
    line_height = font.getbbox("Ay")[3]
    box_height = len(text_lines) * line_height + 10
    box_width = max_line_width + 20

    margin = 10
    x1 = frame_width - box_width - margin
    y1 = margin
    x2 = frame_width - margin
    y2 = margin + box_height

    # Semi-transparent rectangle
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(x1, y1), (x2, y2)], fill=(0, 0, 0, 128))
    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    # Draw text lines
    text_y = y1 + 5
    for line in text_lines:
        draw.text((x1 + 10, text_y), line, font=font, fill=(255, 255, 255))
        text_y += line_height

    return np.array(pil_img)

def record_random_agent(
    output_path="videos/random/random_agent.mp4",
    n_episodes=2,
    max_steps=300,
    width=720,
    height=480
):
    """
    Manually record a random (untrained) agent. Saves a single .mp4 with the specified path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    env = get_env_with_specific_resolution("HumanoidStandup-v4", width, height)

    # **Reset** before we try to render
    obs, info = env.reset()
    first_frame = env.render()
    if first_frame is None:
        print("[DEBUG ERROR] env.render() returned None. Cannot record video.")
        env.close()
        return

    frame_h, frame_w, _ = first_frame.shape
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    step_count = 0
    # We already did the initial reset
    # We'll treat that as episode_num=1's start for the first iteration
    episode_num = 1
    episode_reward = 0
    episode_length = 0

    # We do N episodes, but we handle the first reset outside
    for ep in range(n_episodes):
        # If it's not the very first iteration, do a new reset
        if ep > 0:
            obs, info = env.reset()
            episode_num = ep + 1
            episode_reward = 0
            episode_length = 0

        for t in range(max_steps):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            step_count += 1

            frame = env.render()
            annotated = add_text_overlay(
                frame, 
                episode_num, 
                step_count, 
                episode_reward, 
                episode_length,
                stats_data=None,
                show_stats=True
            )
            out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            if done or truncated:
                print(f"[DEBUG] Random Agent - Episode {episode_num} reward={episode_reward:.2f}")
                break

    out.release()
    env.close()
    print(f"[DEBUG] Random agent video saved to {output_path}")

def record_trained_agent(
    model_path,
    output_path="videos/humanoid_standup-final.mp4",
    n_episodes=5,
    max_steps=300,
    width=720,
    height=480,
    custom_stats=None
):
    """
    Manually record a trained PPO agent, with text overlay. 
    Output saved to a single .mp4 (by default, 'videos/humanoid_standup-final.mp4').
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[DEBUG] Loading model from {model_path}")
    model = PPO.load(model_path)

    env = get_env_with_specific_resolution("HumanoidStandup-v4", width, height)
    # Reset first, THEN render
    obs, info = env.reset()
    first_frame = env.render()
    if first_frame is None:
        print("[DEBUG ERROR] env.render() returned None. Cannot record video.")
        env.close()
        return

    frame_h, frame_w, _ = first_frame.shape
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    step_count = 0
    episode_num = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_num += 1
        episode_reward = 0
        episode_length = 0

        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            step_count += 1

            # If you want to gather stats from the model's logger:
            stats_data = custom_stats.copy() if custom_stats else {}
            if hasattr(model, "logger") and hasattr(model.logger, "name_to_value"):
                for key in ["explained_variance", "learning_rate", "approx_kl", "clip_fraction"]:
                    if key in model.logger.name_to_value:
                        stats_data[key] = model.logger.name_to_value[key]

            frame = env.render()
            annotated = add_text_overlay(
                frame,
                episode_num,
                step_count,
                episode_reward,
                episode_length,
                stats_data=stats_data,
                show_stats=True
            )
            out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            if done or truncated:
                print(f"[DEBUG] Trained Agent - Episode {episode_num}, reward={episode_reward:.2f}")
                break

    out.release()
    env.close()
    print(f"[DEBUG] Trained agent video saved to {output_path}")

def record_checkpoint_agent(
    model_path,
    video_folder="videos/checkpoints/step_XXXXX",
    max_steps=300,
    width=720,
    height=480,
    custom_stats=None
):
    """
    Record exactly 1 episode of the agent from a checkpoint and save to
    'video_folder/humanoid_standup-episode-0.mp4', so that combine_videos.py can detect it.
    """
    os.makedirs(video_folder, exist_ok=True)
    output_path = os.path.join(video_folder, "humanoid_standup-episode-0.mp4")
    print(f"[DEBUG] Recording checkpoint agent -> {output_path}")

    # Same approach as record_trained_agent, but 1 episode
    record_trained_agent(
        model_path=model_path,
        output_path=output_path,
        n_episodes=1,
        max_steps=max_steps,
        width=width,
        height=height,
        custom_stats=custom_stats
    )
