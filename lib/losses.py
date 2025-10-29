import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        return loss



class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def dice_loss(self, pred, target, smooth=1):
        """
        Computes the Dice Loss for binary segmentation.
        Args:
            pred: Tensor of predictions (batch_size, 1, H, W).
            target: Tensor of ground truth (batch_size, 1, H, W).
            smooth: Smoothing factor to avoid division by zero.
        Returns:
            Scalar Dice Loss.
        """
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)
        
        # Calculate intersection and union
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        # Compute Dice Coefficient
        dice = (2. * intersection + smooth) / (union + smooth)
        
        # Return Dice Loss
        return 1 - dice.mean()
    
    
    def forward(self, y_pred, y_true):
        return self.dice_loss(y_pred, y_true)
        raise Exception("Implement this!")

class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def focal_loss(self, inputs, targets, alpha=1, gamma=2):
        # Binary Cross-Entropy loss calculation
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Convert BCE loss to probability
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss  # Apply focal adjustment
        return focal_loss.mean()

    def forward(self, y_pred, y_true):
        return self.focal_loss(y_pred, y_true)
        raise Exception("Implement this!")

class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        # Total Variation Regularization / Contiguity
        regularization = torch.sum(torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])) + \
                         torch.sum(torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, :, 1:, :]))
        return loss + 0.1*regularization

