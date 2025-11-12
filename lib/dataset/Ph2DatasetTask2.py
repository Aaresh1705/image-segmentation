import torch
import os
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
from PIL import Image
import glob
from torchvision import transforms as T
import numpy as np

DATA_PATH = '/dtu/datasets1/02516/PH2_Dataset_images/'

class PH2_Dataset_CSV(torch.utils.data.Dataset):
    def __init__(self, data_path, csv_path, transform=None):
        """
        Load PH2 dataset images along with weak CSV annotations.
        Args:
            data_path: root path containing IMD* folders with images
            csv_path: folder containing CSV annotation files (same filenames as images)
            transform: optional Albumentations transform for the image
        """
        self.transform = transform
        self.image_paths = []
        self.csv_paths = []
        self.data_path = data_path
        self.csv_path = csv_path

        # Scan all patient folders (IMD001, IMD002, ...)
        patient_folders = sorted(glob.glob(os.path.join(data_path, 'IMD*')))
        for folder in patient_folders:
            image_folder = os.path.join(folder, os.path.basename(folder) + '_Dermoscopic_Image')
            imgs = sorted(glob.glob(os.path.join(image_folder, '*.bmp')))

            for img_path in imgs:
                filename = os.path.basename(img_path)
                base = os.path.splitext(filename)[0]
                csv_file = os.path.join(csv_path, base + '.csv')

                # Only include if CSV exists
                if os.path.exists(csv_file):
                    self.image_paths.append(img_path)
                    self.csv_paths.append(csv_file)
                else:
                    print(f"[Warning] Missing CSV for {filename}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # --- Load image ---
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = np.array(image)

        # --- Load CSV with sparse lesion annotations ---
        csv_file = self.csv_paths[idx]
        df = pd.read_csv(csv_file)

        # (x, y, lesion) as a NumPy array of shape [N, 3]
        # Example: [[408,144,1], [326,248,1], [508,43,0], ...]
        points = df[['x', 'y', 'lesion']].values.astype(np.float32)

        # --- Apply transforms (image only) ---
        if self.transform:
            # image = self.transform(image=image)["image"]
            image = self.transform(image)

        return image, points


def datasetPH2(batch_size=8, transform=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Create the dataset
    # dataset = PH2_Dataset_images(transform=transform)
    dataset = PH2_Dataset_CSV(data_path=DATA_PATH, csv_path='/zhome/63/6/222806/Project3/image-segmentation/Task2_dataset', transform=transform)


    # Compute lengths for splits
    total_len = len(dataset)
    train_len = max(1, int(total_len * train_ratio))
    val_len = max(0, int(total_len * val_ratio))
    test_len = total_len - train_len - val_len

    # Split dataset
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader, test_loader), (train_set, val_set, test_set)

#example Usage
# Define transform
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor()
])

# Initialize dataset loaders
(train_loader, val_loader, test_loader), (train_set, val_set, test_set) = datasetPH2(
    batch_size=4,
    transform=transform
)

# 🔹 One-liner test: print batch image shape and point tensor length
for X, Y in train_loader:
    print(f"Images: {X.shape}, Points per sample: {[y.shape for y in Y]}")
    break
