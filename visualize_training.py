import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import glob

def ensure_directory_structure(log_folder):
    """Ensure the log folder has the right structure for Monitor files"""
    # Make sure the directory exists
    os.makedirs(log_folder, exist_ok=True)
    
    # Check if there are any monitor files in the directory
    monitor_files = glob.glob(os.path.join(log_folder, "*.monitor.csv"))
    
    # If it's empty but the parent directory has monitor files, copy them
    if len(monitor_files) == 0:
        parent_dir = os.path.dirname(log_folder)
        parent_monitor_files = glob.glob(os.path.join(parent_dir, "*.monitor.csv"))
        
        if len(parent_monitor_files) > 0:
            import shutil
            for file in parent_monitor_files:
                shutil.copy(file, log_folder)
                print(f"Copied {file} to {log_folder}")
    
    return glob.glob(os.path.join(log_folder, "*.monitor.csv"))

def plot_training_results(log_folder, title="Learning Curve", save_to_file=True, show_plot=True):
    """
    Plot the training reward over time.
    
    Args:
        log_folder: Path to the logs folder
        title: Title of the plot
        save_to_file: Whether to save the plot to a file
        show_plot: Whether to display the plot
    """
    # Ensure directory structure and find monitor files
    monitor_files = ensure_directory_structure(log_folder)
    
    if len(monitor_files) == 0:
        print(f"No monitor files found in {log_folder}")
        return
    
    # Load results
    try:
        data = load_results(log_folder)
        if len(data) == 0:
            print(f"No data found in {log_folder}")
            return
    except Exception as e:
        print(f"Error loading results: {e}")
        
        # Try to load individual files instead
        all_data = []
        for file in monitor_files:
            try:
                file_data = load_results(file)
                if len(file_data) > 0:
                    all_data.append(file_data)
            except:
                pass
        
        if not all_data:
            print("Could not load any data from monitor files")
            return
        
        # Combine data
        import pandas as pd
        data = pd.concat(all_data)
    
    # Plot training reward
    try:
        x, y = ts2xy(data, 'timesteps')
        
        if len(y) == 0:
            print("No reward data found")
            return
            
        # Apply a rolling average to smooth the curve
        window = max(1, len(y) // 20)  # 5% of total data
        y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
        x_smooth = x[len(x)-len(y_smooth):]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        if len(y) > 1:  # Only plot raw if we have more than one point
            plt.plot(x, y, 'b-', alpha=0.3, label='Raw rewards')
        
        if len(y_smooth) > 1:  # Only plot smooth if we have more than one point
            plt.plot(x_smooth, y_smooth, 'r-', label='Smoothed rewards')
        
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Rewards')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        # Add training statistics text box
        stats_text = f"Total Episodes: {len(data)}\n"
        stats_text += f"Total Timesteps: {x[-1]:,}\n"
        if len(y) > 0:
            stats_text += f"Final Reward: {y[-1]:.2f}\n"
            stats_text += f"Max Reward: {np.max(y):.2f}\n"
            stats_text += f"Mean Reward: {np.mean(y):.2f}"
        
        plt.figtext(0.13, 0.15, stats_text, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save the plot
        if save_to_file:
            os.makedirs("plots", exist_ok=True)
            plt.savefig(os.path.join("plots", "learning_curve.png"), dpi=150)
            print(f"Plot saved to plots/learning_curve.png")
        
        if show_plot:
            plt.show()
            
    except Exception as e:
        print(f"Error creating plot: {e}")

def plot_all_metrics(log_folder, show_plot=True):
    """
    Plot multiple training metrics if available.
    
    Args:
        log_folder: Path to the logs folder
        show_plot: Whether to display the plot
    """
    try:
        # Load TensorBoard data if available
        from tensorboard.backend.event_processing import event_accumulator
        
        # Find all event files
        event_files = glob.glob(os.path.join(log_folder, "**", "events.out.tfevents.*"), recursive=True)
        
        if not event_files:
            print("No TensorBoard event files found")
            return
            
        # Load each event file
        for event_file in event_files:
            try:
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                
                # Get available tags (metrics)
                tags = ea.Tags()['scalars']
                
                if not tags:
                    continue
                    
                # Create a multi-panel figure
                num_tags = len(tags)
                fig, axes = plt.subplots(num_tags, 1, figsize=(12, 4 * num_tags), sharex=True)
                
                # If only one tag, make axes iterable
                if num_tags == 1:
                    axes = [axes]
                
                # Plot each metric
                for i, tag in enumerate(tags):
                    events = ea.Scalars(tag)
                    steps = [event.step for event in events]
                    values = [event.value for event in events]
                    
                    axes[i].plot(steps, values)
                    axes[i].set_title(tag)
                    axes[i].grid(True)
                    
                    if i == num_tags - 1:
                        axes[i].set_xlabel('Steps')
                
                plt.tight_layout()
                
                # Save figure
                os.makedirs("plots", exist_ok=True)
                metric_filename = f"metrics_{os.path.basename(event_file)}.png"
                plt.savefig(os.path.join("plots", metric_filename), dpi=150)
                print(f"Metrics plot saved to plots/{metric_filename}")
                
                if show_plot:
                    plt.show()
                    
            except Exception as e:
                print(f"Error processing event file {event_file}: {e}")
    
    except ImportError:
        print("TensorBoard not installed. Cannot plot additional metrics.")

if __name__ == "__main__":
    # Try both monitor directories
    try:
        plot_training_results("logs/monitor")
    except Exception as e:
        print(f"Error with logs/monitor: {e}")
        try:
            plot_training_results("logs")
        except Exception as e:
            print(f"Error with logs: {e}")
    
    # Try to plot additional metrics from TensorBoard logs
    try:
        plot_all_metrics("logs")
    except Exception as e:
        print(f"Error plotting additional metrics: {e}")