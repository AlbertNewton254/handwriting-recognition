"""
plots.py - Plotting functions for training metrics
"""

import os
import matplotlib.pyplot as plt


def save_loss_plot(epochs_list, train_losses, val_losses, save_dir):
    """
    Generate and save loss plot

    epochs_list: List of epoch numbers
    train_losses: List of training losses
    val_losses: List of validation losses
    save_dir: Directory to save the plot

    plot_path: Path to the saved plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs_list, val_losses, label='Val Loss', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, "loss_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def save_cer_plot(epochs_list, val_cers, save_dir):
    """
    Generate and save CER plot

    epochs_list: List of epoch numbers
    val_cers: List of validation CER values
    save_dir: Directory to save the plot

    plot_path: Path to the saved plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, val_cers, label='Val CER', marker='o', markersize=3, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('CER (%)')
    plt.title('Validation Character Error Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, "cer_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path
