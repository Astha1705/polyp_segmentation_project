import numpy as np
import matplotlib.pyplot as plt

# Load saved metrics
data = np.load("metrics.npz")
dice_scores, iou_scores = data['dice_scores'], data['iou_scores']

# Function to print summary stats
def print_stats(name, scores):
    print(f"\n=== {name} Stats ===")
    print(f"Mean:   {np.mean(scores):.4f}")
    print(f"Median: {np.median(scores):.4f}")
    print(f"Min:    {np.min(scores):.4f}")
    print(f"Max:    {np.max(scores):.4f}")
    print(f"Std:    {np.std(scores):.4f}")

print_stats("Dice Coefficient", dice_scores)
print_stats("IoU", iou_scores)

# Function to plot histogram
def plot_hist(scores, title, color, xlabel):
    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=30, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Plot histograms
plot_hist(dice_scores, "Distribution of Dice Coefficients", 'skyblue', "Dice Coefficient")
plot_hist(iou_scores, "Distribution of IoU Scores", 'lightgreen', "IoU")

# Function to plot boxplot
def plot_box(scores, title, color, xlabel):
    plt.figure(figsize=(6, 4))
    plt.boxplot(scores, vert=False, patch_artist=True,
                boxprops=dict(facecolor=color, color='black'),
                medianprops=dict(color='red'))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()

# Plot boxplots
plot_box(dice_scores, "Boxplot of Dice Coefficients", 'skyblue', "Dice Coefficient")
plot_box(iou_scores, "Boxplot of IoU Scores", 'lightgreen', "IoU")

# Scatter plot: Dice vs IoU
plt.figure(figsize=(6, 6))
plt.scatter(dice_scores, iou_scores, alpha=0.5, color='purple')
plt.title("Scatter plot: Dice vs IoU")
plt.xlabel("Dice Coefficient")
plt.ylabel("IoU")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
