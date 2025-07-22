import torch
import numpy as np
from unet import UNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load model
model = UNet().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Dataset definition
class PolypDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            image, mask = self.transform(image), self.transform(mask)
        return image, mask

# Prepare test data
test_image_paths = sorted(glob("Kvasir-SEG/images/*.jpg"))
test_mask_paths  = sorted(glob("Kvasir-SEG/masks/*.jpg"))
print(f"[INFO] Found {len(test_image_paths)} test images and {len(test_mask_paths)} masks")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

test_loader = DataLoader(
    PolypDataset(test_image_paths, test_mask_paths, transform=transform),
    batch_size=4, shuffle=False
)

# Metrics
def compute_metric(pred, target, mode='dice', epsilon=1e-6):
    pred, target = pred.view(-1), target.view(-1)
    intersection = (pred * target).sum()
    if mode == 'dice':
        return (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
    elif mode == 'iou':
        union = pred.sum() + target.sum() - intersection
        return (intersection + epsilon) / (union + epsilon)
    else:
        raise ValueError("Unsupported mode")

# Evaluate
dice_scores, iou_scores = [], []

with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = (model(images) > 0.5).float()

        for pred, target in zip(outputs, masks):
            dice_scores.append(compute_metric(pred, target, mode='dice').item())
            iou_scores.append(compute_metric(pred, target, mode='iou').item())

# Results
print(f"[RESULT] Mean Dice Coefficient: {np.mean(dice_scores):.4f}")
print(f"[RESULT] Mean IoU: {np.mean(iou_scores):.4f}")

# Save metrics
np.savez("metrics.npz", dice_scores=dice_scores, iou_scores=iou_scores)
print("[INFO] Saved per-image metrics to metrics.npz")
