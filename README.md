# Reinforcement Learning Video Creator

This project trains reinforcement learning agents using the Gymnasium library and creates high-quality videos for YouTube content. It focuses on the HumanoidStandup environment from MuJoCo, recording the progress of training and automatically combining all videos into a polished final product.

## Features

- **Training with progress tracking**: Train RL agents with detailed progress metrics
- **High-quality video recording**: Capture agent performance at specified intervals
- **Custom video resolution**: Set specific dimensions for your videos (default: 720x480)
- **Overlay statistics**: Videos include episode rewards, steps, and training metrics
- **Automatic video combination**: Creates a single video showing training progression
- **Long training support**: Run for hours to create thorough training videos

## Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

Run a short training session to test everything:

```bash
python main.py --timesteps 50000 --record-interval 10000
```

Run a long training session for YouTube content:

```bash
python train_long.py
```

## Running Options

### Main Training Script

```bash
python main.py [options]
```

Options:
- `--timesteps`: Total training steps (default: 1,000,000)
- `--record-interval`: Video recording interval (default: 100,000)
- `--video-width`: Video width in pixels (default: 720)
- `--video-height`: Video height in pixels (default: 480)
- `--no-final-video`: Skip creating combined video
- `--output-video`: Path for combined video (default: training_video.mp4)
- `--video-every`: Only record every N checkpoints

### Long Training Launcher

```bash
python train_long.py [options]
```

Options:
- `--timesteps`: Training steps (default: 5,000,000)
- `--record-interval`: Video recording interval (default: 250,000)
- `--video-every`: Only record every N checkpoints (default: 2)
- `--width`: Video width (default: 720)
- `--height`: Video height (default: 480)
- `--output`: Output video filename
- `--est-fps`: Estimated training FPS for time estimation

### Video Combining Script

```bash
python combine_videos.py [options]
```

Options:
- `--output`: Output video path
- `--fps`: Frames per second (default: 30)
- `--no-titles`: Disable title screens
- `--max-duration`: Maximum output duration in seconds (default: 15 minutes)
- `--videos-dir`: Directory containing videos

## Example Workflow for YouTube Content

1. **Run a long training session** (5-6 hours):
   ```bash
   python train_long.py --timesteps 5000000 --record-interval 250000
   ```

2. **Review the automatic combined video** created at the end

3. **Optional: Customize the combined video**:
   ```bash
   python combine_videos.py --output custom_video.mp4 --max-duration 900
   ```

## Video Format

The final combined video includes:
1. Title screens introducing each section
2. Random agent performance (untrained)
3. Training progress at regular intervals
4. Final trained agent performance
5. Training summary

All videos include overlay text showing:
- Episode number
- Current step
- Episode reward
- Training statistics (when available)
- Timestamp

## Project Structure

- `main.py`: Main training and recording script
- `train_long.py`: Script for long training sessions
- `record_videos.py`: Video recording with overlay
- `combine_videos.py`: Combines videos into final product
- `visualize_training.py`: Training progress plots
- `models/`: Saved model checkpoints
- `videos/`: Recorded agent performances
- `plots/`: Training progress visualizations

## Customization Tips

1. **Adjust training hyperparameters** in main.py for different learning behaviors
2. **Change video content** by modifying combine_videos.py
3. **Customize overlay information** in record_videos.py
4. **Change environment** by replacing "HumanoidStandup-v4" with other MuJoCo environments

## Troubleshooting

- **Memory issues**: Reduce batch size or recording frequency
- **Slow training**: Check hardware acceleration, reduce network size
- **Missing videos**: Ensure directories exist and check permissions
- **Video problems**: Install required codecs (h264)

## Requirements

See `requirements.txt` for detailed dependencies.