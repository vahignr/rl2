import os
import glob
import re
import argparse
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips

def extract_step_number(path):
    """Extract step number from path for proper ordering"""
    match = re.search(r'step_(\d+)', path)
    if match:
        return int(match.group(1))
    return 0

def get_video_files(source_dir):
    """Get all video files from a directory and subdirectories"""
    # Get all video files
    video_files = []
    for ext in ['mp4', 'avi', 'mov']:
        video_files.extend(glob.glob(os.path.join(source_dir, '**', f'*.{ext}'), recursive=True))
    
    # Filter out empty files
    video_files = [f for f in video_files if os.path.getsize(f) > 0]
    
    return video_files

def combine_videos(output_path, random_videos=None, checkpoint_videos=None, 
                  final_videos=None, fps=30, add_titles=True, max_duration=15*60):
    """
    Combine videos from different training stages into a single video.
    Simplified version without title screens but maintains parameter compatibility.
    
    Note: add_titles parameter is maintained for compatibility but not used.
    """
    clips = []
    
    # Process random agent videos
    if random_videos:
        try:
            random_clips = [VideoFileClip(vid) for vid in random_videos[:2]]  # Limit to 2 random videos
            if random_clips:
                random_combined = concatenate_videoclips(random_clips)
                clips.append(random_combined)
                print("Added random agent videos")
        except Exception as e:
            print(f"Error processing random agent videos: {e}")
    
    # Process checkpoint videos - select a subset for long trainings
    if checkpoint_videos:
        try:
            # Sort checkpoints by step number
            checkpoint_videos.sort(key=extract_step_number)
            
            # For very long trainings, pick checkpoints at regular intervals
            if len(checkpoint_videos) > 10:
                # Select about 10 clips evenly spaced
                indices = np.linspace(0, len(checkpoint_videos)-1, 10).astype(int)
                selected_checkpoints = [checkpoint_videos[i] for i in indices]
            else:
                selected_checkpoints = checkpoint_videos
            
            print(f"Selected {len(selected_checkpoints)} checkpoint videos from {len(checkpoint_videos)} total")
            
            # Create clips for each checkpoint
            for i, vid_path in enumerate(selected_checkpoints):
                try:
                    clip = VideoFileClip(vid_path)
                    clips.append(clip)
                    print(f"Added checkpoint video: {vid_path}")
                except Exception as e:
                    print(f"Error with checkpoint video {vid_path}: {e}")
        except Exception as e:
            print(f"Error processing checkpoint videos: {e}")
    
    # Process final model videos
    if final_videos:
        try:
            final_clips = [VideoFileClip(vid) for vid in final_videos[:3]]  # Limit to 3 final videos
            if final_clips:
                final_combined = concatenate_videoclips(final_clips)
                clips.append(final_combined)
                print("Added final agent videos")
        except Exception as e:
            print(f"Error processing final agent videos: {e}")
    
    if not clips:
        print("No video clips found to combine!")
        return
    
    # Combine all clips
    print(f"Combining {len(clips)} video segments")
    final_clip = concatenate_videoclips(clips)
    
    # Check if the video is too long and needs to be sped up
    total_duration = final_clip.duration
    print(f"Initial video duration: {total_duration:.1f} seconds")
    
    speed_factor = 1.0
    if total_duration > max_duration:
        speed_factor = total_duration / max_duration
        print(f"Video is too long, speeding up by {speed_factor:.2f}x")
        final_clip = final_clip.speedx(speed_factor)
        print(f"New duration: {final_clip.duration:.1f} seconds")
    
    # Write final video
    print(f"Writing video to {output_path}...")
    final_clip.write_videofile(output_path, fps=fps, threads=4, 
                              codec='libx264', audio=False)
    print(f"Combined video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Combine reinforcement learning training videos')
    parser.add_argument('--output', type=str, default='combined_training.mp4',
                        help='Output video file path')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for output video')
    parser.add_argument('--no-titles', action='store_true',
                        help='Disable title screens between sections')
    parser.add_argument('--max-duration', type=int, default=15*60,
                        help='Maximum duration of output video in seconds (default: 15 minutes)')
    parser.add_argument('--videos-dir', type=str, default='videos',
                        help='Root directory containing all videos')
    
    args = parser.parse_args()
    
    # Get video files from different categories
    random_dir = os.path.join(args.videos_dir, 'random')
    checkpoints_dir = os.path.join(args.videos_dir, 'checkpoints')
    
    random_videos = get_video_files(random_dir) if os.path.exists(random_dir) else []
    checkpoint_videos = get_video_files(checkpoints_dir) if os.path.exists(checkpoints_dir) else []
    
    # Get final videos but exclude directories to prevent errors
    all_files = get_video_files(args.videos_dir)
    final_videos = [f for f in all_files 
                   if 'random' not in f and 'checkpoints' not in f and os.path.isfile(f)]
    
    print(f"Found {len(random_videos)} random agent videos")
    print(f"Found {len(checkpoint_videos)} checkpoint videos")
    print(f"Found {len(final_videos)} final agent videos")
    
    try:
        combine_videos(
            args.output,
            random_videos=random_videos,
            checkpoint_videos=checkpoint_videos,
            final_videos=final_videos,
            fps=args.fps,
            add_titles=not args.no_titles,  # Maintain parameter for compatibility
            max_duration=args.max_duration
        )
    except Exception as e:
        print(f"Error creating combined video: {e}")
        print("You can still view the individual video files in the 'videos' directory.")

if __name__ == "__main__":
    main()