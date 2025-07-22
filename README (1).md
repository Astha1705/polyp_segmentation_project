
# Polyp Segmentation with U-Net

This project performs semantic segmentation of gastrointestinal (GI) polyps using a U-Net architecture on the Kvasir-SEG dataset.

It includes:
- Model training and evaluation
- Visualization of predictions
- Exploratory Data Analysis (EDA) of segmentation metrics (Dice coefficient and IoU)

---

## Project Structure
polyp_segmentation_project/
└── Kvasir-SEG/
    ├── images/
    └── masks/
    ├── model.pth                     # Saved trained U-Net model
    ├── metrics.npz                   # Saved per-image Dice and IoU scores
    ├── evaluate.py                   # Script to compute and save metrics
    ├── visualize_predictions.py      # Script to view sample predictions
    ├── eda_metrics.py                # Script for EDA: plots and statistics
    ├── unet.py                       # U-Net model definition
    ├── utils.py                      # Helper functions (if needed)
    ├── main.py                       # (Optional) training script
    └── README.md                     # Project documentation
```

---

## Dataset

- **Name:** Kvasir-SEG  
- **Description:** 1000 polyp images with segmentation masks collected from real endoscopic examinations.  
- **Link:** [https://datasets.simula.no/kvasir-seg/](https://datasets.simula.no/kvasir-seg/)

Dataset should be organized as:
```
Kvasir-SEG/images/*.jpg
Kvasir-SEG/masks/*.jpg
```

---

## How to Run

**1) Install dependencies**

Install required Python packages:

```bash
pip install torch torchvision numpy matplotlib pillow
```

---

**2) Evaluate the trained model**

Compute the mean Dice coefficient and IoU over your test data:

```bash
python evaluate.py
```

This script will:
- Load `model.pth`
- Run inference on the test dataset
- Print mean Dice and IoU
- Save per-image metrics to `metrics.npz`

---

**3) Visualize predictions**

To display a few sample predictions side by side with ground truth masks:

```bash
python visualize_predictions.py
```

---

**4) Perform exploratory data analysis**

Plot histograms, boxplots, scatter plots, and summary statistics of the Dice and IoU scores:

```bash
python eda_metrics.py
```

---

## Model

- Architecture: U-Net (see `unet.py`)
- Input images are resized (e.g., to 256×256)
- Model is trained to predict binary segmentation masks

---

## Evaluation Metrics

- **Dice coefficient:** Measures overlap between predicted and true masks. Higher values indicate better performance.
- **IoU (Intersection over Union):** Ratio of intersection to union of predicted and true masks. Also known as Jaccard index.

---

## Possible Improvements

- Use a deeper backbone or pretrained encoder for U-Net
- Apply data augmentation to improve generalization
- Experiment with different learning rates and optimizers
- Visualize failure cases to better understand model limitations

---

## License

This project is intended for educational and research purposes.
