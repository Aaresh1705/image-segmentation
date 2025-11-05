# pip install torchsummary
import torch
import torch.optim as optim
from time import time

import albumentations as A
from albumentations.pytorch import ToTensorV2

from lib.dataset import datasetDRIVE, datasetPH2
from lib.model import EncDecNet, DilatedNet, UNet
from lib.losses import BCELoss, DiceLoss, FocalLoss, BCELoss_TotalVariation

# Dataset
size = 128
transform = A.Compose([A.Resize(size, size),
                        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),  # <- converts to float32 for albumentations
                        ToTensorV2(),
                        ])
  # transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()]))

batch_size = 6
# (train_loader, val_loader, test_loader), (trainset, valset, testset) = datasetDRIVE(batch_size=batch_size, transform=transform)
(train_loader, val_loader, test_loader), (trainset, valset, testset) = datasetPH2(batch_size=batch_size, transform=transform)
# IMPORTANT NOTE: There is no validation set provided here, but don't forget to
# have one for the project

print(f"Loaded {len(trainset)} training images")
print(f"Loaded {len(valset)} val images")
print(f"Loaded {len(testset)} test images")

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncDecNet().to(device)
#model = UNet().to(device) # TODO
#model = UNet2().to(device) # TODO
#model = DilatedNet().to(device) # TODO
#summary(model, (3, 256, 256))
learning_rate = 0.001
opt = optim.Adam(model.parameters(), learning_rate)

loss_fn = BCELoss()
# loss_fn = DiceLoss() # TODO
# loss_fn = FocalLoss() # TODO
# loss_fn = BCELoss_TotalVariation() # TODO
epochs = 20

# Training loop
X_test, Y_test = next(iter(test_loader))

model.train()  # train mode
for epoch in range(epochs):
    tic = time()
    print(f'* Epoch {epoch+1}/{epochs}')

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

    # IMPORTANT NOTE: It is a good practice to check performance on a
    # validation set after each epoch.
    #model.eval()  # testing mode
    #Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
    print(f' - loss: {avg_loss}')

# Save the model
#torch.save(model, ....)
print("Training has finished!")
