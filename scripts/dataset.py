import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class AlzheimerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory (e.g., train or test folder).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Map labels to integers (AD -> 0, MCI -> 1, CN -> 2)
        self.label_to_idx = {"AD": 0, "MCI": 1, "CN": 2}
        
        # Collect all images and their labels
        for label_dir in os.listdir(root_dir):  # AD, MCI, CN directories
            label_path = os.path.join(root_dir, label_dir)
            if os.path.isdir(label_path) and label_dir in self.label_to_idx:  # Ensure label exists in mapping:
                for img_file in os.listdir(label_path):
                    if img_file.endswith(".png"):
                        img_path = os.path.join(label_path, img_file)
                        self.samples.append((img_path, label_dir))  # (image_path, label)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB format
        label_idx = self.label_to_idx[label]  # Convert label to integer
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx
    

train_dir = "/Users/halimat/Documents/alzheimer_project/CNN_recreation/data/processed/train"
train_dataset = AlzheimerDataset(train_dir)
for i in range(5):
    img, label = train_dataset[i]
    print(f"Image: {train_dataset.samples[i][0]}, Label: {label}")