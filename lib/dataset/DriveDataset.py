from glob import glob
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T


class DRIVE(Dataset):
    def __init__(self,
                 root_dir='/dtu/datasets1/02516/DRIVE/',
                 transform=None):
        self.transform = transform

        train_images = sorted(glob(os.path.join(root_dir, 'training/images/*.tif')))
        train_masks = sorted(glob(os.path.join(root_dir, 'training/mask/*.gif')))

        test_images = sorted(glob(os.path.join(root_dir, 'test/images/*.tif')))
        test_masks = sorted(glob(os.path.join(root_dir, 'test/mask/*.gif')))

        images = train_images + test_images
        masks = train_masks + test_masks

        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = T.ToTensor()(image)
            mask = T.ToTensor()(mask)

        return image, mask


def datasetDRIVE(batch_size=64, transform=None, split=(0.8, 0.1, 0.1)):
    assert sum(split) == 1.0

    dataset = DRIVE(transform=transform)

    n = len(dataset)
    n_train = int(split[0] * n)
    n_val = int(split[1] * n)
    n_test = n - n_train - n_val

    trainset, valset, testset = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader, test_loader), (trainset, valset, testset)
