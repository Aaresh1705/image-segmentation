from datetime import date, datetime
import matplotlib.pyplot as plt
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
from lib.model.EncDecModel import EncDecNet as EncDec
from lib.model.EncDecModel import EncDecNet
from lib.model.DilatedNetModel import DilatedNet
from lib.model.UNetModel import UNet, UNet2 #, UNet2
from lib.losses import BCELoss, BCELoss_PositiveWeights, DiceLoss, FocalLoss, BCELoss_TotalVariation
from measure import evaluate_model
from lib import all_losses

# Dataset
size = 128
transform = A.Compose([A.Resize(size, size),
                        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),  # <- converts to float32 for albumentations
                        ToTensorV2(),
                        ])
  # transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()]))

batch_size = 6
# (train_loader, val_loader, test_loader), (trainset, valset, testset) = datasetDRIVE(batch_size=batch_size, transform=transform)
(train_loader, val_loader, test_loader), (trainset, valset, testset) = Ph2Dataset.datasetPH2(batch_size=batch_size, transform=transform)
# IMPORTANT NOTE: There is no validation set provided here, but don't forget to
# have one for the project

print(f"Loaded {len(trainset)} training images")
print(f"Loaded {len(valset)} val images")
print(f"Loaded {len(testset)} test images")

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncDec().to(device)
#model = UNet().to(device) # TODO
#model = UNet2().to(device) # TODO
#model = DilatedNet().to(device) # TODO
#summary(model, (3, 256, 256))
learning_rate = 0.0001
opt = optim.Adam(model.parameters(), learning_rate)

# loss_fn = BCELoss()
loss_fn = DiceLoss() 
# loss_fn = FocalLoss()
# loss_fn = BCELoss_TotalVariation() 

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
    for X_batch, y_true in train_loader:
        X_batch = X_batch.to(device)
        y_true = y_true.to(device)

        # set parameter gradients to zero
        opt.zero_grad()

        # forward
        y_pred = model(X_batch)
        # IMPORTANT NOTE: Check whether y_pred is normalized or unnormalized
        # and whether it makes sense to apply sigmoid or softmax.
        loss = loss_fn(y_pred, y_true)  # forward-pass
        loss.backward()  # backward-pass
        opt.step()  # update weights

        # calculate metrics to show the user
        avg_loss += loss / len(train_loader)
        
        

    model.eval()
    avg_loss_val = 0
    for X_val, y_val in val_loader:
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        y_pred = model(X_val)
        loss = loss_fn(y_pred, y_val)
        avg_loss_val += loss / len(val_loader)

    model.eval()
    avg_loss_val = 0
    for X_val, y_val in val_loader:
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        y_pred = model(X_val)
        loss = loss_fn(y_pred, y_val)
        avg_loss_val += loss / len(val_loader)

    # IMPORTANT NOTE: It is a good practice to check performance on a
    # validation set after each epoch.
    # model.eval()  # testing mode
    # Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
    print(f' -  train loss: {avg_loss:.3f}')
    print(f' -  val loss:   {avg_loss_val:.3f}')
    #model.eval()  # testing mode
    #Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
    print(f' - loss: {avg_loss}')
    # save the loss curve in a file with the date and time of running
    #also save the plot
    
    summary = evaluate_model(model, val_loader)

    with open(f"loss_curve_final_ph2.txt", "a") as f:
        f.write(f"{epoch+1},{avg_loss},{summary}\n")







if __name__ == "__main__":
    # Dataset
    size = 128
    transform = A.Compose([A.Resize(size, size),
                           A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                           # <- converts to float32 for albumentations
                           ToTensorV2(),
                           ])
    # transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()]))

    batch_size = 6
    (train_loader, val_loader, test_loader), (trainset, valset, testset) = datasetPH2(batch_size=batch_size,
                                                                                        transform=transform)

    print(f"Loaded {len(trainset)} training images")
    print(f"Loaded {len(valset)} val images")
    print(f"Loaded {len(testset)} test images")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = UNet().to(device) # TODO
    # model = UNet2().to(device) # TODO
    # model = DilatedNet().to(device) # TODO

    learning_rate = 0.0001
    epochs = 20

    for loss_function in all_losses:
        model = UNet2(n_channels=3, n_classes=1).to(device)
        opt = optim.Adam(model.parameters(), learning_rate)
        loss_function = loss_function()
        # print loss function name
        print(f"Using Loss function: {loss_function.name}")
        # If loss function is BCELoss_PositiveWeights, set find the ratio of positive to negative pixels in the training set
        if loss_function.name == "BCELoss_PositiveWeights":
            pos_pixels = 0
            neg_pixels = 0
            for _, masks in train_loader:
                pos_pixels += torch.sum(masks == 1).item()
                neg_pixels += torch.sum(masks == 0).item()
            pos_weight = pos_pixels / (pos_pixels + neg_pixels)
            loss_function = BCELoss_PositiveWeights(pos_weight=pos_weight)
            print(f"Positive weight set to: {pos_weight:.3f}")
            
        for epoch in range(epochs):
            print(f"Using {loss_function.name}")
            print(f'------==={{{epoch+1:>2}}}===------')
            train(model, train_loader, loss_function, opt)
            print(evaluate_model(model, val_loader))
            model.train()

    # Save the model
    # torch.save(model, ....)
    summary(model, (3, 256, 256))
    print("Training has finished!")