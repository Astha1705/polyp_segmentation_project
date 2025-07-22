import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import PolypDataset, get_transforms
from unet import UNet

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# --- Paths ---
image_dir, mask_dir = "Kvasir-SEG/images", "Kvasir-SEG/masks"

# --- File names ---
image_filenames = sorted(os.listdir(image_dir))
mask_filenames  = sorted(os.listdir(mask_dir))

# --- Train-validation split ---
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_filenames, mask_filenames, test_size=0.2, random_state=42
)

# --- Transforms ---
transform = get_transforms(image_size=256)

# --- Datasets & DataLoaders ---
train_dataset = PolypDataset(image_dir, mask_dir, transform=transform, files=train_imgs)
val_dataset   = PolypDataset(image_dir, mask_dir, transform=transform, files=val_imgs)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False)

# --- Model, loss, optimizer ---
model = UNet(n_channels=3, n_classes=1).to(device)
bce_loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return 1 - ((2 * intersection + smooth) / (union + smooth)).mean()

def total_loss(pred, target):
    return bce_loss(pred, target) + dice_loss(pred, target)

# --- Training loop ---
def train_model(max_epochs=30, patience=3):
    best_val_loss, patience_counter = float("inf"), 0

    for epoch in range(1, max_epochs+1):
        # Training
        model.train()
        train_losses = []
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = total_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                val_losses.append(total_loss(preds, masks).item())

        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch {epoch:02}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss, patience_counter = avg_val_loss, 0
            torch.save(model.state_dict(), "model.pth")
            print("[INFO] Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("[INFO] Early stopping triggered")
                break

# --- Run ---
if __name__ == "__main__":
    train_model()
