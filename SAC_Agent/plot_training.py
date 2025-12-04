#!/usr/bin/env python3
"""
Plot training metrics from CSV files generated during training.

Usage:
    python plot_training.py logs/model_name_timestamp.csv
    python plot_training.py logs/  # Plot all CSV files in logs directory
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Get script directory for relative paths
script_dir = Path(__file__).parent


def plot_training_metrics(csv_path, output_dir=None, show_plot=True):
    """
    Plot training metrics from a CSV file.

    Args:
        csv_path: Path to CSV file or directory containing CSV files
        output_dir: Directory to save plots (default: same as CSV file)
        show_plot: Whether to display plots interactively
    """
    csv_path = Path(csv_path)

    # Handle directory input
    if csv_path.is_dir():
        csv_files = list(csv_path.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {csv_path}")
            return
        print(f"Found {len(csv_files)} CSV file(s). Plotting all...")
        for csv_file in csv_files:
            plot_single_file(csv_file, output_dir, show_plot)
    elif csv_path.is_file():
        plot_single_file(csv_path, output_dir, show_plot)
    else:
        print(f"Error: {csv_path} is not a valid file or directory")
        sys.exit(1)


def plot_single_file(csv_file, output_dir=None, show_plot=True):
    """Plot metrics from a single CSV file."""
    print(f"Loading metrics from {csv_file}...")
    df = pd.read_csv(csv_file)

    if len(df) == 0:
        print(f"Warning: {csv_file} is empty")
        return

    # Determine output directory
    if output_dir is None:
        output_dir = csv_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Training Metrics: {csv_file.stem}", fontsize=16, fontweight="bold")

    # 1. Reward and Success Rate
    ax1 = plt.subplot(2, 3, 1)
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(df["episode"], df["reward"], "b-", label="Reward", linewidth=1.5)
    line2 = ax1_twin.plot(
        df["episode"], df["success_rate"] * 100, "r-", label="Success Rate (%)", linewidth=1.5
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color="b")
    ax1_twin.set_ylabel("Success Rate (%)", color="r")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1_twin.tick_params(axis="y", labelcolor="r")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Reward & Success Rate")
    # Combined legend
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    # 2. Q-Losses
    ax2 = plt.subplot(2, 3, 2)
    if "q1_loss" in df.columns and df["q1_loss"].notna().any():
        ax2.plot(df["episode"], df["q1_loss"], "g-", label="Q1 Loss", linewidth=1.5)
    if "q2_loss" in df.columns and df["q2_loss"].notna().any():
        ax2.plot(df["episode"], df["q2_loss"], "orange", label="Q2 Loss", linewidth=1.5)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.set_title("Q-Network Losses")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Actor Loss
    ax3 = plt.subplot(2, 3, 3)
    if "actor_loss" in df.columns and df["actor_loss"].notna().any():
        ax3.plot(df["episode"], df["actor_loss"], "purple", label="Actor Loss", linewidth=1.5)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Loss")
    ax3.set_title("Actor Loss")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Alpha (Temperature)
    ax4 = plt.subplot(2, 3, 4)
    if "alpha" in df.columns and df["alpha"].notna().any():
        ax4.plot(df["episode"], df["alpha"], "brown", label="Alpha", linewidth=1.5)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Alpha")
    ax4.set_title("Temperature Parameter (Alpha)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Episode Length
    ax5 = plt.subplot(2, 3, 5)
    if "episode_length" in df.columns and df["episode_length"].notna().any():
        ax5.plot(df["episode"], df["episode_length"], "teal", label="Episode Length", linewidth=1.5)
        # Add moving average
        window = min(50, len(df) // 10)
        if window > 1:
            ma = df["episode_length"].rolling(window=window, center=True).mean()
            ax5.plot(df["episode"], ma, "r--", label=f"MA({window})", linewidth=1, alpha=0.7)
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Steps")
    ax5.set_title("Episode Length")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Motion Metrics (if available)
    ax6 = plt.subplot(2, 3, 6)
    has_motion_metrics = False
    if "mean_jerk" in df.columns and df["mean_jerk"].notna().any():
        ax6.plot(df["episode"], df["mean_jerk"], "c-", label="Mean Jerk", linewidth=1.5)
        has_motion_metrics = True
    if "rms_acc" in df.columns and df["rms_acc"].notna().any():
        ax6.plot(df["episode"], df["rms_acc"], "m-", label="RMS Acceleration", linewidth=1.5)
        has_motion_metrics = True
    if "collisions" in df.columns and df["collisions"].notna().any():
        ax6_twin = ax6.twinx()
        motion_col = df["mean_jerk"] if "mean_jerk" in df.columns else df["rms_acc"]
        line1 = ax6.plot(df["episode"], motion_col, "c-", label="Motion Quality", linewidth=1.5)
        line2 = ax6_twin.plot(df["episode"], df["collisions"], "r-", label="Collisions", linewidth=1.5)
        ax6_twin.set_ylabel("Collisions", color="r")
        ax6_twin.tick_params(axis="y", labelcolor="r")
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax6.legend(lines, labels, loc="upper left")
        has_motion_metrics = True
    else:
        if has_motion_metrics:
            ax6.legend()

    if has_motion_metrics:
        ax6.set_xlabel("Episode")
        ax6.set_ylabel("Motion Quality")
        ax6.set_title("Motion Quality Metrics")
    else:
        ax6.text(
            0.5, 0.5, "No motion metrics available",
            ha="center", va="center", transform=ax6.transAxes
        )
        ax6.set_title("Motion Quality Metrics")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_filename = output_dir / f"{csv_file.stem}_training_curves.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_filename}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # Print summary statistics
    print("\n" + "=" * 80)
    print(f"Summary Statistics for {csv_file.stem}")
    print("=" * 80)
    print(f"Total Episodes: {len(df)}")
    print(f"Final Reward: {df['reward'].iloc[-1]:.3f}")
    print(f"Best Reward: {df['reward'].max():.3f}")
    print(f"Mean Reward (last 100): {df['reward'].tail(100).mean():.3f}")
    print(f"Final Success Rate: {df['success_rate'].iloc[-1]:.2%}")
    print(f"Mean Success Rate (last 100): {df['success_rate'].tail(100).mean():.2%}")
    if "episode_length" in df.columns:
        print(f"Mean Episode Length: {df['episode_length'].mean():.1f}")
    if "q1_loss" in df.columns and df["q1_loss"].notna().any():
        print(f"Final Q1 Loss: {df['q1_loss'].iloc[-1]:.4f}")
    if "actor_loss" in df.columns and df["actor_loss"].notna().any():
        print(f"Final Actor Loss: {df['actor_loss'].iloc[-1]:.4f}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_training.py logs/sac_finetune_20240101_120000.csv
  python plot_training.py logs/  # Plot all CSV files in logs directory
  python plot_training.py logs/ --output plots/  # Save plots to different directory
  python plot_training.py logs/ --no-show  # Don't display plots, just save them
        """,
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file or directory containing CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save plots (default: same as CSV file location)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots interactively, just save them",
    )
    args = parser.parse_args()

    # Resolve paths relative to script directory if not absolute
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        # Try relative to script directory first, then current working directory
        script_csv_path = script_dir / args.csv_path
        if script_csv_path.exists() or script_csv_path.is_dir():
            csv_path = script_csv_path
        else:
            csv_path = Path(args.csv_path).resolve()

    output_dir = None
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_dir = str(script_dir / args.output)
        else:
            output_dir = args.output

    plot_training_metrics(str(csv_path), output_dir, show_plot=not args.no_show)


if __name__ == "__main__":
    main()
