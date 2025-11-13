import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_line(line):
    metrics = {}
    epoch_match = re.search(r'epochs:\s*(\d+)', line)
    metrics["epoch"] = int(epoch_match.group(1)) if epoch_match else None

    # Train and validation loss
    train_match = re.search(r'train_loss:\s*([\deE\.\-]+)', line)
    val_match = re.search(r'val_loss:\s*([\deE\.\-]+)', line)
    metrics["train_loss"] = float(train_match.group(1)) if train_match else None
    metrics["val_loss"] = float(val_match.group(1)) if val_match else None

    # Other metrics
    summaries = re.findall(r'(\w+):\s*([-\deE\.]+)', line.split("summary:")[-1])
    for key, value in summaries:
        metrics[key] = float(value)
    return metrics


def plot_metrics(filename):
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    data = pd.DataFrame([parse_line(l) for l in lines])

    # --- Detect blocks automatically (each block starts with epoch = 1)
    block_ids = []
    current_block = 0
    for i, epoch in enumerate(data["epoch"]):
        if epoch == 1 and i > 0:
            current_block += 1
        block_ids.append(current_block)
    data["loss_func_id"] = block_ids

    # Custom names for each loss function
    loss_labels = ["BCELoss_PositiveWeights", "BCELoss", "FocalLoss", "BCELoss_TotalVariation"]
    data["loss_func_name"] = data["loss_func_id"].map(lambda i: loss_labels[i] if i < len(loss_labels) else f"Loss_{i}")

    metrics = [c for c in data.columns if c not in ["epoch", "loss_func_id", "loss_func_name"]]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for name, subset in data.groupby("loss_func_name"):
            plt.plot(subset["epoch"], subset[metric], label=name, marker='o')
        plt.title(metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"losses_DRIVE/{metric}_plot.png", dpi=150)
        plt.close()
        print(f"Saved {metric}_plot.png")

    print("\n✅ All metric plots saved successfully!")


if __name__ == "__main__":
    plot_metrics("loss_curve_final_DRIIVE.txt")
