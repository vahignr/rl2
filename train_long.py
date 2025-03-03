#!/usr/bin/env python
"""
Run long training sessions for YouTube videos.
This uses the manual recorder references in `main.py`.
"""
import os
import subprocess
import argparse
import time
from datetime import datetime, timedelta

def time_format(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def estimate_training_time(timesteps, fps):
    seconds = timesteps / fps
    return time_format(seconds)

def main():
    parser = argparse.ArgumentParser(description='Run long training for YouTube video creation')
    parser.add_argument('--timesteps', type=int, default=5_000_000, help='Training timesteps')
    parser.add_argument('--record-interval', type=int, default=250_000, help='Recording interval')
    parser.add_argument('--video-every', type=int, default=2, help='Only record video every N intervals')
    parser.add_argument('--width', type=int, default=720, help='Video width')
    parser.add_argument('--height', type=int, default=480, help='Video height')
    parser.add_argument('--output', type=str, default="humanoid_standup_training.mp4", help='Output video path')
    parser.add_argument('--est-fps', type=float, default=2000, help='Estimated training FPS')

    args = parser.parse_args()
    est_time = estimate_training_time(args.timesteps, args.est_fps)
    current_time = datetime.now()
    estimated_finish = current_time + timedelta(seconds=args.timesteps / args.est_fps)

    print(f"=== Long Training Configuration ===")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Recording interval: {args.record_interval:,}")
    print(f"Video resolution: {args.width}x{args.height}")
    print(f"Estimated training time: {est_time}")
    print(f"Estimated completion: {estimated_finish.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 35)

    confirmation = input("This will start a long training session. Continue? (y/n): ")
    if confirmation.lower() != 'y':
        print("Training cancelled.")
        return

    start_time = time.time()
    cmd = [
        "python", "main.py",
        "--timesteps", str(args.timesteps),
        "--record-interval", str(args.record_interval),
        "--video-width", str(args.width),
        "--video-height", str(args.height),
        "--output-video", args.output,
        "--video-every", str(args.video_every)
    ]
    print("\nStarting training...")
    process = subprocess.Popen(cmd)
    try:
        process.wait()
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {time_format(elapsed)}")
        print(f"Final video saved to: {args.output}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Stopping...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        elapsed = time.time() - start_time
        print(f"Training stopped after {time_format(elapsed)}")

if __name__ == "__main__":
    main()
