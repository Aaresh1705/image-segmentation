import datetime
from datetime import datetime
import torch
import torch.optim as optim
from torchsummary import summary
from torch.nn import functional as F
from measure import evaluate_model
from torch.utils.data import DataLoader 
import albumentations as A
from albumentations.pytorch import ToTensorV2

from lib.dataset import datasetDRIVE
from lib.model.UNetModel import UNet, UNet2
from lib.losses import BCELoss, DiceLoss, FocalLoss, BCELoss_TotalVariation
from lib import all_losses


def train(model, train_loader, loss_fn, opt):
    model.train()  # train mode
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        y_pred = F.sigmoid(y_pred)

        loss = loss_fn(y_pred, y_val)
        avg_loss_val += loss / len(val_loader)

    # IMPORTANT NOTE: It is a good practice to check performance on a
    # validation set after each epoch.
    # model.eval()  # testing mode
    # Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
    summary = evaluate_model(model, val_loader)

    with open(f"loss_curve_final_DRIIVE.txt", "a") as f:
        f.write(f"epochs: {epoch+1}, train_loss: {avg_loss}, val_loss: {avg_loss_val}, summary: {summary}\n")

    return avg_loss, avg_loss_val




if __name__ == "__main__":
    # Dataset
    size = 128
    
    transform = A.Compose([
        A.Resize(height=size, width=size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomCrop(height=size, width=size, p=0.8),
        ToTensorV2()
    ])
    # transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()]))

    batch_size = 6
    (train_loader, val_loader, test_loader), (trainset, valset, testset) = datasetDRIVE(batch_size=batch_size,
                                                                                        transform=transform)

    print(f"Loaded {len(trainset)} training images")
    print(f"Loaded {len(valset)} val images")
    print(f"Loaded {len(testset)} test images")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = EncDecNet().to(device)
    # model = UNet2().to(device) # TODO
    # model = DilatedNet().to(device) # TODO
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    learning_rate = 1e-3
    epochs = 20
    for loss_function in all_losses:
        model = UNet2(n_channels=3, n_classes=1).to(device) 
        # model = EncDecNet().to(device)
        opt = optim.Adam(model.parameters(), learning_rate)
        loss_function = loss_function()
        print(loss_function.name)

        best_loss = float('inf')
        for epoch in range(epochs):
            train_loss, val_loss = train(model, train_loader, loss_function, opt)
            print(f'------==={{{epoch+1:>2}}}===------')
            print(loss_function.name)
            print(f' -  train loss: {train_loss:.3f}')
            print(f' -  val loss:   {val_loss:.3f}')
            print(evaluate_model(model, val_loader))
            model.train()
            if val_loss < best_loss:
                best_loss = val_loss
                print('Updating best model')
                # Save the model
                torch.save(model, f'models/DRIVE/{model.name}.{loss_function.name}.pth')
        #Write to file "Final scores" after all epochs are done
        final_summary_train = evaluate_model(model, train_loader)
        final_summary_test = evaluate_model(model, test_loader)
        final_summary_val = evaluate_model(model, val_loader)
        final_summary = f"Train: {final_summary_train}, Val: {final_summary_val}, Test: {final_summary_test}"
        with open(f"final_scores/final_scores_drive.txt", "a") as f:
            f.write(f"Model: {model.name}, Loss Function: {loss_function.name}, time: {now} Summary: {final_summary}\n")
        print(f"Final evaluation on test set: {final_summary}")

    summary(model, (3, 256, 256))   
    print("Training has finished!")

    #Visualize some test results by showing an image and its predicted mask
    model.eval()
    import matplotlib.pyplot as plt
    X_test, y_test = next(iter(test_loader))
    X_test = X_test.to(device)
    y_pred = (model(X_test)).detach().cpu()
    y_test = y_test.unsqueeze(1)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    print("y_pred:", y_pred.shape)
    num_images = 4
    for i in range(num_images):
        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.title("Input Image")
        plt.imshow(X_test[i].permute(1,2,0).cpu().squeeze())
        plt.axis('off')
        plt.subplot(1,3,2)  
        plt.title("Predicted Mask")
        plt.imshow(y_pred[i,0].squeeze(), cmap='gray')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.title("Ground Truth Mask")
        plt.imshow(y_test[i,0].squeeze(), cmap='gray')
        plt.axis('off')
        plt.savefig(f"test_predictions_DRIVE_{i}.png", dpi=150)
    print("Test predictions visualized!")    # plot_metrics("loss_curve_final_DRIIVE.txt")