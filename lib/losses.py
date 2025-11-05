import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    """Binary Cross entropy loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        return loss



class DiceLoss(nn.Module):
    """Dice loss function

    """
    def __init__(self):
        super().__init__()

    def dice_loss(self, pred, target, smooth=1):

        nominator = torch.mean(2 * pred * target + smooth)
        denominator = torch.mean(pred + target) + smooth
        # Return Dice Loss
        return 1 - (nominator / denominator)
    
    
    def forward(self, y_pred, y_true):
        if y_true.ndim == 3:
            y_true = y_true.unsqueeze(1)
        return self.dice_loss(y_pred, y_true)


class FocalLoss(nn.Module):
    """Focal loss function

    Args:
        alpha (float): Weighting factor for the class
        gamma (float): Focusing parameter
        reduction (str): Reduction method
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Match dimensions if necessary
        if y_true.ndim == 3:
            y_true = y_true.unsqueeze(1)

        # Compute binary cross-entropy with logits 
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")

        # Compute pt = exp(-bce_loss) = predicted probability of the true class
        pt = torch.exp(-bce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # Reduce to scalar
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class BCELoss_TotalVariation(nn.Module):
    """Bincary cross entropy with Total Variance regularization

    Args:
        tv_weight (float): Weighting factor for the total variance term
    """
    def __init__(self, tv_weight=0.1):
        super().__init__()
        self.tv_weight = tv_weight

    def forward(self, y_pred, y_true):
        # Ensure target has channel dimension
        if y_true.ndim == 3:
            y_true = y_true.unsqueeze(1)

        #  Binary Cross-Entropy 
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

        #  Apply sigmoid for contiguity term
        y_prob = torch.sigmoid(y_pred)

        #  Compute total variation (horizontal + vertical differences)
        diff_h = torch.abs(y_prob[:, :, :, 1:] - y_prob[:, :, :, :-1])
        diff_v = torch.abs(y_prob[:, :, 1:, :] - y_prob[:, :, :-1, :])
        tv_loss = torch.sum(diff_h) + torch.sum(diff_v)

        #  Normalize by batch size to avoid scaling with image size
        tv_loss = tv_loss / y_pred.size(0)

        #  Combine both
        loss = bce_loss + self.tv_weight * tv_loss
        return loss

