import numpy as np
import matplotlib.pyplot as plt

# Load metrics
data = np.load("metrics.npz")
metrics = [("Dice Coefficient", data['dice_scores'], 'skyblue'),
           ("IoU", data['iou_scores'], 'lightgreen')]

# Plot distributions
for title, scores, color in metrics:
    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=30, color=color, edgecolor='black')
    plt.title(f"Distribution of {title}s")
    plt.xlabel(title)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
