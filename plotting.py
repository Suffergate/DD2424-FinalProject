import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from config import PLOT_SAVE_DIR


def load_plot_data(model_type, hidden_dim):
    """Load the saved plot data for a specific model"""
    plot_data_filename = f"{model_type}_hidden{hidden_dim}_plot_data.pkl"
    plot_data_path = os.path.join(PLOT_SAVE_DIR, "data", plot_data_filename)

    if os.path.exists(plot_data_path):
        with open(plot_data_path, "rb") as f:
            return pickle.load(f)
    else:
        return None


def plot_comparison_losses(
    model_types, hidden_dims, smoothing_window=100, save_path=None
):

    plt.figure(figsize=(12, 8))

    colors = ["blue", "red", "green"]
    markers = ["o", "s", "^"]

    for i, (model_type, hidden_dim) in enumerate(zip(model_types, hidden_dims)):
        data = load_plot_data(model_type, hidden_dim)
        if data is None:
            continue

        # Get the data
        train_losses = data.get("train_losses", [])
        val_losses = data.get("val_losses", [])
        val_iterations = data.get("val_iterations", [])

        # Smooth training losses
        if len(train_losses) > smoothing_window:
            iterations = np.arange(1, len(train_losses) + 1)
            smooth_losses = np.convolve(
                train_losses, np.ones(smoothing_window) / smoothing_window, mode="valid"
            )
            smooth_iterations = iterations[smoothing_window - 1 :]

            # Plot smoothed training loss
            plt.plot(
                smooth_iterations,
                smooth_losses,
                "-",
                color=colors[i],
                label=f"{model_type.upper()} (h={hidden_dim}) - Training Loss",
                linewidth=1.5,
                alpha=0.7,
            )

        # Plot validation loss
        plt.plot(
            val_iterations,
            val_losses,
            "-",
            color=colors[i],
            marker=markers[i],
            label=f"{model_type.upper()} (h={hidden_dim}) - Validation Loss",
            linewidth=2,
            markersize=5,
            markevery=max(1, len(val_iterations) // 20),
        )

    plt.title("Loss Comparison LSTM2 Model Hidden layers", fontsize=16)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.ylim(1, 2.5)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss comparison plot saved to {save_path}")
    else:
        plt.show()


def generate_comparison_plots():
    """Generate the comparison plots for the three baseline models"""
    # Model configurations
    model_types = ["lstm2", "lstm2", "lstm2"]
    hidden_dims = [64, 128, 256]

    # Create output directory
    comparison_dir = os.path.join(PLOT_SAVE_DIR, "model_comparisons")
    os.makedirs(comparison_dir, exist_ok=True)

    # Generate loss comparison plot
    loss_plot_path = os.path.join(comparison_dir, "lstm2_loss_comparison.png")
    plot_comparison_losses(model_types, hidden_dims, save_path=loss_plot_path)


if __name__ == "__main__":
    generate_comparison_plots()
