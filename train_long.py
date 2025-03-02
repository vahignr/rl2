#!/usr/bin/env python
"""
Script for running long training sessions for creating YouTube videos.
This will train for several hours and generate a nice final video.
"""

import os
import subprocess
import argparse
import time
from datetime import datetime, timedelta

def time_format(seconds):
    """Format seconds into a human-readable time string"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def estimate_training_time(timesteps, fps):
    """Estimate training time based on steps and estimated FPS"""
    seconds = timesteps / fps
    return time_format(seconds)

def main():
    parser = argparse.ArgumentParser(description='Run long training for YouTube video creation')
    parser.add_argument('--timesteps', type=int, default=5_000_000,
                        help='Total training timesteps (default: 5,000,000)')
    parser.add_argument('--record-interval', type=int, default=250_000,
                        help='Interval for recording videos (default: 250,000)')
    parser.add_argument('--video-every', type=int, default=2,
                        help='Only record video every N intervals to save space (default: 2)')
    parser.add_argument('--width', type=int, default=720,
                        help='Video width (default: 720)')
    parser.add_argument('--height', type=int, default=480,
                        help='Video height (default: 480)')
    parser.add_argument('--output', type=str, default="humanoid_standup_training.mp4",
                        help='Output video path (default: humanoid_standup_training.mp4)')
    parser.add_argument('--est-fps', type=float, default=2000,
                        help='Estimated training FPS for time estimation (default: 2000)')
    
    args = parser.parse_args()
    
    # Show estimated training time
    est_time = estimate_training_time(args.timesteps, args.est_fps)
    current_time = datetime.now()
    estimated_finish = current_time + timedelta(seconds=args.timesteps/args.est_fps)
    
    print(f"=== Long Training Configuration ===")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Recording interval: {args.record_interval:,}")
    print(f"Video resolution: {args.width}x{args.height}")
    print(f"Estimated training time: {est_time}")
    print(f"Estimated completion: {estimated_finish.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 35)
    
    # Confirm before starting
    confirmation = input("This will start a long training session. Continue? (y/n): ")
    if confirmation.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Start timer
    start_time = time.time()
    
    # Build the command
    cmd = [
        "python", "main.py",
        "--timesteps", str(args.timesteps),
        "--record-interval", str(args.record_interval),
        "--video-width", str(args.width),
        "--video-height", str(args.height),
        "--output-video", args.output,
        "--video-every", str(args.video_every)
    ]
    
    # Execute the command
    print("\nStarting training...")
    process = subprocess.Popen(cmd)
    
    try:
        process.wait()
        
        # Training finished
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {time_format(elapsed)}")
        print(f"Final video saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Stopping...")
        process.terminate()
        
        # Wait for process to terminate
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        
        elapsed = time.time() - start_time
        print(f"Training stopped after {time_format(elapsed)}")

if __name__ == "__main__":
    main()