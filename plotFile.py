import pandas as pd
import matplotlib.pyplot as plt
import re
import sys

def parse_line(line):
    """Extract numeric values from one line."""
    parts = line.strip().split(',')
    epoch = int(parts[0])
    loss = float(parts[1])
    
    metrics = {'Epoch': epoch, 'Loss': loss}
    for p in parts[2:]:
        # Extract key and value like "IoU: 0.23"
        match = re.match(r'\s*([^:]+):\s*([-\deE\.]+)', p)
        if match:
            key, val = match.groups()
            metrics[key.strip()] = float(val)
    return metrics


def plot_metrics(filename):
    # Read and parse file
    with open(filename, 'r') as f:
        lines = [l for l in f.readlines() if l.strip()]
    
    # Convert to dataframe
    data = pd.DataFrame([parse_line(l) for l in lines])
    
    # Split into groups of 20 epochs per loss function
    n_losses = len(data) // 20
    loss_names = ["BCELoss_PositiveWeights","BCELoss", "FocalLoss", "BCELoss_TotalVariation"]
    
    metrics = [c for c in data.columns if c not in ['Epoch']]
    
    # Plot each metric
    for metric in metrics:
        plt.figure(figsize=(8,5))
        for i in range(n_losses):
            subset = data.iloc[i*20:(i+1)*20]
            plt.plot(subset['Epoch'], subset[metric], label=loss_names[i])
        
        plt.title(metric)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"losses_ph2/{metric}_plot.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_losses.py <datafile>")
        sys.exit(1)
    plot_metrics(sys.argv[1])
