import csv
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """
    Logger for training sessions that saves both to console and file.
    Also exports metrics to CSV for easy visualization.
    """

    def __init__(self, model_name, log_dir="logs", log_to_file=True):
        """
        Initialize training logger.

        Args:
            model_name: Name of the model being trained
            log_dir: Directory to save log files
            log_to_file: Whether to save logs to file
        """
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.log_to_file = log_to_file
        self.start_time = datetime.now()

        # Create log directory
        if self.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Create log file with timestamp
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            log_filename = f"{model_name}_{timestamp}.log"
            self.log_file = self.log_dir / log_filename

            # Create CSV file for metrics
            csv_filename = f"{model_name}_{timestamp}.csv"
            self.csv_file = self.log_dir / csv_filename
            self.csv_writer = None
            self.csv_file_handle = None
            self._init_csv()

            # Write header
            with open(self.log_file, "w") as f:
                f.write("=" * 80 + "\n")
                f.write(f"Training Log: {model_name}\n")
                f.write(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Log File: {self.log_file}\n")
                f.write(f"CSV File: {self.csv_file}\n")
                f.write("=" * 80 + "\n\n")

    def _init_csv(self):
        """Initialize CSV file with headers."""
        self.csv_file_handle = open(self.csv_file, "w", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_file_handle,
            fieldnames=[
                "episode",
                "reward",
                "success_rate",
                "episode_length",
                "q1_loss",
                "q2_loss",
                "actor_loss",
                "alpha",
                "alpha_loss",
                "mean_jerk",
                "rms_acc",
                "collisions",
            ],
        )
        self.csv_writer.writeheader()
        self.csv_file_handle.flush()

    def log(self, message, level="INFO"):
        """
        Log a message to both console and file.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, etc.)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"

        # Print to console
        print(log_message)

        # Write to file
        if self.log_to_file:
            with open(self.log_file, "a") as f:
                f.write(log_message + "\n")

    def log_episode(self, episode, reward, success_rate, loss_info=None, metrics=None):
        """
        Log episode information and export to CSV.

        Args:
            episode: Episode number
            reward: Episode reward
            success_rate: Success rate
            loss_info: Dictionary of loss information
            metrics: Dictionary of additional metrics
        """
        message = f"Episode {episode:5d} | Reward: {reward:8.3f} | Success: {success_rate:.2%}"
        if loss_info:
            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in loss_info.items()])
            message += f" | {loss_str}"
        if metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            message += f" | {metrics_str}"

        self.log(message)

        # Export to CSV
        if self.log_to_file and self.csv_writer:
            row = {
                "episode": episode,
                "reward": reward,
                "success_rate": success_rate,
                "episode_length": metrics.get("length", 0) if metrics else 0,
                "q1_loss": loss_info.get("q1_loss", 0.0) if loss_info else 0.0,
                "q2_loss": loss_info.get("q2_loss", 0.0) if loss_info else 0.0,
                "actor_loss": loss_info.get("actor_loss", 0.0) if loss_info else 0.0,
                "alpha": loss_info.get("alpha", 0.0) if loss_info else 0.0,
                "alpha_loss": loss_info.get("alpha_loss", 0.0) if loss_info else 0.0,
                "mean_jerk": metrics.get("mean_jerk", 0.0) if metrics else 0.0,
                "rms_acc": metrics.get("rms_acc", 0.0) if metrics else 0.0,
                "collisions": metrics.get("collisions", 0.0) if metrics else 0.0,
            }
            self.csv_writer.writerow(row)
            self.csv_file_handle.flush()  # Ensure data is written immediately

    def log_checkpoint(self, episode, mean_reward, filepath):
        """
        Log checkpoint save.

        Args:
            episode: Episode number
            mean_reward: Mean reward for checkpoint
            filepath: Path to saved checkpoint
        """
        message = (
            f"Checkpoint saved at Episode {episode} | "
            f"Mean Reward: {mean_reward:.3f} | "
            f"File: {filepath}"
        )
        self.log(message, level="CHECKPOINT")

    def log_summary(self, total_episodes, best_reward, final_reward, total_time):
        """
        Log training summary.

        Args:
            total_episodes: Total number of episodes
            best_reward: Best reward achieved
            final_reward: Final reward
            total_time: Total training time
        """
        self.log("=" * 80)
        self.log("Training Summary")
        self.log("=" * 80)
        self.log(f"Model: {self.model_name}")
        self.log(f"Total Episodes: {total_episodes}")
        self.log(f"Best Reward: {best_reward:.3f}")
        self.log(f"Final Reward: {final_reward:.3f}")
        self.log(f"Total Time: {total_time:.2f} seconds")
        self.log(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 80)

    def close(self):
        """Close the logger and CSV file."""
        if self.log_to_file:
            if self.csv_file_handle:
                self.csv_file_handle.close()
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            self.log(f"Training completed. Duration: {duration:.2f} seconds")
            self.log(f"Log file saved to: {self.log_file}")
            self.log(f"CSV metrics file saved to: {self.csv_file}")
