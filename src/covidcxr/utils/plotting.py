from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history, output_path: str | Path | None = None) -> None:
    keys = history.history.keys()
    loss_key = "loss"
    val_loss_key = "val_loss"
    acc_key = "binary_accuracy" if "binary_accuracy" in keys else "accuracy"
    val_acc_key = f"val_{acc_key}"

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=160)

    ax[0].plot(history.history.get(loss_key, []), label="train")
    if val_loss_key in history.history:
        ax[0].plot(history.history[val_loss_key], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(history.history.get(acc_key, []), label="train")
    if val_acc_key in history.history:
        ax[1].plot(history.history[val_acc_key], label="val")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    plt.close(fig)
