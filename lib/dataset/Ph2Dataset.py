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

class PH2_Dataset_images(torch.utils.data.Dataset):
    def __init__(self, transform):
        """
        Load the entire PH2 dataset.
        """
        self.transform = transform
        self.image_paths = []
        self.label_paths = []

        # Scan all patient folders
        patient_folders = sorted(glob.glob(os.path.join(DATA_PATH, 'IMD*')))
        for folder in patient_folders:
            image_folder = os.path.join(folder, os.path.basename(folder) + '_Dermoscopic_Image')
            label_folder = os.path.join(folder, os.path.basename(folder) + '_lesion')

            # Get all images and masks
            imgs = sorted(glob.glob(os.path.join(image_folder, '*.bmp')))
            masks = sorted(glob.glob(os.path.join(label_folder, '*.bmp')))

            # Ensure number of images matches number of masks
            assert len(imgs) == len(masks), f"Mismatch in {folder}"

            self.image_paths.extend(imgs)
            self.label_paths.extend(masks)

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx]).convert('L')  # mask as grayscale

        #Needs to be numpy for albumentations to work
        image = np.array(image)
        mask = np.array(label)
        
        # Apply transform if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask

def datasetPH2(batch_size=8, transform=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Create the dataset
    dataset = PH2_Dataset_images(transform=transform)

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

"""example Usage

# Define transforms (optional)
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# Call the dataset loader function
(train_loader, test_loader, val_loader), (train_set, test_set, val_set) = datasetPH2(
    batch_size=4,
    transform=transform
)

# Print dataset info
print(f"Total train samples: {len(train_set)}")
print(f"Total validation samples: {len(val_set)}")
print(f"Total test samples: {len(test_set)}")

# Check a single batch
for X, Y in train_loader:
    print(f"Image batch shape: {X.shape}")  # should be (batch_size, 3, H, W)
    print(f"Mask batch shape: {Y.shape}")   # should be (batch_size, 1, H, W)
    print(f"Image dtype: {X.dtype}, Mask dtype: {Y.dtype}")
    break  # just check first batch
"""