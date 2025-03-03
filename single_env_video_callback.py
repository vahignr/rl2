import os
import numpy as np
import datetime
from PIL import Image, ImageDraw, ImageFont

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import VecEnv

try:
    import moviepy as mpy
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

class SingleEnvVideoRecorderCallback(BaseCallback):
    """
    A custom callback that runs a single environment for the entire training,
    capturing each step into a single .mp4 (via MoviePy) with text overlay.

    By default, we stop training after `max_episodes` or when total timesteps are reached,
    whichever comes first.

    Requirements:
    - Your environment must be a single-env VecEnv (like DummyVecEnv with 1 env).
    - That env must have render_mode="rgb_array".
    - `moviepy` installed (or some way to write frames).
    """

    def __init__(
        self,
        video_path="training_single_env.mp4",
        fps=30,
        max_episodes=100,
        verbose=1
    ):
        super().__init__(verbose=verbose)
        self.video_path = video_path
        self.fps = fps
        self.max_episodes = max_episodes

        self.frames = []  # we'll store raw frames in a list, then write to .mp4 at end
        self.episode_num = 0
        self.global_step = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _init_callback(self) -> None:
        """Called once at the start of training."""
        os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
        if not MOVIEPY_AVAILABLE:
            raise ImportError(
                "moviepy is not installed. Install it via `pip install moviepy` to record a single environment video."
            )

    def _on_training_start(self) -> None:
        """Called before the first rollout starts."""
        if self.verbose > 0:
            print(f"[DEBUG] SingleEnvVideoRecorderCallback: Writing video to {self.video_path}")
        # We can do an initial reset/render
        env = self.training_env.envs[0]  # single environment
        obs, _ = env.reset()
        first_frame = env.render()
        if first_frame is not None:
            annotated = self._annotate_frame(first_frame)
            self.frames.append(annotated)
        else:
            if self.verbose > 0:
                print("[DEBUG] Warning: env.render() returned None at the start.")

    def _on_step(self) -> bool:
        """Called at every environment step."""
        env = self.training_env.envs[0]
        self.global_step += 1

        # We can see if the environment just ended an episode via info
        info = self.locals.get("infos", [{}])[0]  # stable-baselines passes "infos"
        maybe_ep_info = info.get("episode")  # if using Monitor wrapper, "episode": {"r":..., "l":...}
        if maybe_ep_info:
            # That means the previous step ended an episode
            self.episode_num += 1
            # For logging or debugging
            if self.verbose > 0:
                print(
                    f"[DEBUG] Episode {self.episode_num} ended. Reward={maybe_ep_info['r']:.2f}, length={maybe_ep_info['l']}"
                )
            # If we've hit max_episodes, stop
            if self.episode_num >= self.max_episodes:
                if self.verbose > 0:
                    print(f"[DEBUG] Reached max_episodes={self.max_episodes}. Stopping training.")
                return False

        # Now we render the env at each step
        frame = env.render()
        if frame is not None:
            annotated = self._annotate_frame(frame)
            self.frames.append(annotated)
        else:
            if self.verbose > 0:
                print("[DEBUG] Warning: env.render() returned None in _on_step.")

        return True  # True => keep training

    def _on_training_end(self) -> None:
        """Called once training is done or interrupted."""
        if self.verbose > 0:
            print(f"[DEBUG] Finalizing single-env video with {len(self.frames)} frames at {self.fps} fps...")

        if not self.frames:
            print("[DEBUG] No frames collected. Not creating a video.")
            return

        clip = mpy.ImageSequenceClip(self.frames, fps=self.fps)
        clip.write_videofile(self.video_path, codec="libx264", audio=False)
        print(f"[DEBUG] Single environment video saved to {self.video_path}")

    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Add text overlay with the current episode number, global step, timestamp, etc.
        We do not have partial-episode reward unless we track it ourselves. 
        Here we just show the last finished episode reward from info.
        """
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        # Attempt fonts
        try:
            font = ImageFont.truetype("Arial", 14)
        except:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()

        text_lines = [
            f"Episode: {max(1,self.episode_num+1)}",  # if we're mid-episode, it's (episode_num+1)
            f"Global Step: {self.global_step}",
        ]

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text_lines.append(f"Time: {current_time}")

        # bounding box for top-right
        frame_h, frame_w = pil_img.height, pil_img.width
        line_widths = [draw.textlength(line, font=font) for line in text_lines]
        max_line_width = max(line_widths) if line_widths else 0
        line_height = font.getbbox("Ay")[3]
        box_height = len(text_lines)*line_height + 10
        box_width = max_line_width + 20

        margin = 10
        x1 = frame_w - box_width - margin
        y1 = margin
        x2 = frame_w - margin
        y2 = margin + box_height

        # Semi-transparent rectangle
        overlay = Image.new('RGBA', pil_img.size, (0,0,0,0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([(x1, y1),(x2,y2)], fill=(0,0,0,128))
        pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_img)

        text_y = y1+5
        for line in text_lines:
            draw.text((x1+10, text_y), line, font=font, fill=(255,255,255))
            text_y += line_height

        return np.array(pil_img)

