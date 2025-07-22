import torch
from unet import UNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

# Setup
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
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# Load data
test_image_paths = sorted(glob("Kvasir-SEG/images/*.jpg"))
test_mask_paths  = sorted(glob("Kvasir-SEG/masks/*.jpg"))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

test_dataset = PolypDataset(test_image_paths, test_mask_paths, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Visualize predictions
num_samples = 5  # number of examples to show

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        outputs = model(images)
        outputs = (outputs > 0.5).float()

        for i in range(images.size(0)):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            mask = masks[i].cpu().squeeze().numpy()
            pred = outputs[i].cpu().squeeze().numpy()

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(pred, cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

            num_samples -= 1
            if num_samples == 0:
                break
        if num_samples == 0:
            break
