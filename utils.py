import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Custom Dataset Class
class PolypDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, file_list=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        if file_list:
            self.image_filenames, self.mask_filenames = zip(*file_list)
        else:
            self.image_filenames = sorted(os.listdir(image_dir))
            self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()
        return image, mask

# Preprocessing Transforms
def get_transforms(image_size=256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
