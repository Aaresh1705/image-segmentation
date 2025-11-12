from datetime import date, datetime
import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.optim as optim
from torchsummary import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

from lib.dataset import datasetDRIVE
from lib.dataset import Ph2Dataset
from lib.dataset import datasetPH2
from lib.dataset import datasetPH2Task2
from lib.model.EncDecModel import EncDecNet as EncDec
from lib.model.EncDecModel import EncDecNet
from lib.model.DilatedNetModel import DilatedNet
from lib.model.UNetModel import UNet #, UNet2
from lib.losses import BCELoss, DiceLoss, FocalLoss, BCELoss_TotalVariation
from lib.lossesTask2 import WeakPointBCELoss  # <-- new weak annotation loss

# Dataset
size = 128
transform = A.Compose([A.Resize(size, size),
                        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),  # <- converts to float32 for albumentations
                        ToTensorV2(),
                        ])
batch_size = 6
(train_loader, val_loader, test_loader), (trainset, valset, testset) = Ph2Dataset.datasetPH2(batch_size=batch_size, transform=transform)

print(f"Loaded {len(trainset)} training images")
print(f"Loaded {len(valset)} val images")
print(f"Loaded {len(testset)} test images")

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncDec().to(device)
learning_rate = 0.0001
opt = optim.Adam(model.parameters(), learning_rate)

# --- Use WeakPointBCELoss ---
loss_fn = WeakPointBCELoss()  

epochs = 20

# Training loop
X_test, Y_test = next(iter(test_loader))

model.train()  # train mode
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
for epoch in range(epochs):
    print(f'* Epoch {epoch+1}/{epochs}')

def train(model, train_loader, loss_fn, opt):
    model.train()  # train mode
    avg_loss = 0
    for X_batch, points_batch in train_loader:
        X_batch = X_batch.to(device)
        # Convert list of points to tensors on device
        y_true_points = [torch.tensor(points, device=device, dtype=torch.float32) for points in points_batch]

        # set parameter gradients to zero
        opt.zero_grad()

        # forward
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_true_points)  # forward-pass with weak annotations
        loss.backward()  # backward-pass
        opt.step()  # update weights

        # accumulate average loss
        avg_loss += loss.item() / len(train_loader)
        
    # Validation
    model.eval()
    avg_loss_val = 0
    with torch.no_grad():
        for X_val, points_val in val_loader:
            X_val = X_val.to(device)
            y_val_points = [torch.tensor(points, device=device, dtype=torch.float32) for points in points_val]

            y_pred_val = model(X_val)
            loss_val = loss_fn(y_pred_val, y_val_points)
            avg_loss_val += loss_val.item() / len(val_loader)

    print(f' -  train loss: {avg_loss:.3f}')
    print(f' -  val loss:   {avg_loss_val:.3f}')

    # save the loss curve in a file with the date and time of running
    with open(f"loss_curve_{now}.txt", "a") as f:
        f.write(f"{epoch+1},{avg_loss},{avg_loss_val}\n")

import matplotlib.pyplot as plt
epochs_list = list(range(1, epochs+1))
losses_train, losses_val = [], []
with open(f"loss_curve_{now}.txt", "r") as f:
    for line in f:
        epoch_num, train_loss, val_loss = line.strip().split(",")
        losses_train.append(float(train_loss))
        losses_val.append(float(val_loss))

plt.plot(epochs_list, losses_train, label="Train Loss")
plt.plot(epochs_list, losses_val, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.savefig(f"loss_curve_{now}.png")
plt.close()

if __name__ == "__main__":
    # Dataset
    size = 128
    transform = A.Compose([A.Resize(size, size),
                           A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                           ToTensorV2(),
                           ])
    batch_size = 6
    (train_loader, val_loader, test_loader), (trainset, valset, testset) = datasetPH2(batch_size=batch_size,
                                                                                        transform=transform)

    print(f"Loaded {len(trainset)} training images")
    print(f"Loaded {len(valset)} val images")
    print(f"Loaded {len(testset)} test images")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncDecNet().to(device)
    summary(model, (3, 256, 256))

    learning_rate = 0.001
    epochs = 20

    # Use WeakPointBCELoss here as well
    opt = optim.Adam(model.parameters(), learning_rate)
    loss_function = WeakPointBCELoss()

    for epoch in range(epochs):
        print(f'------==={{{epoch+1:>2}}}===------')
        train(model, train_loader, loss_function, opt)

    print("Training has finished!")
