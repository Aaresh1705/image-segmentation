# Here, you can load the predicted segmentation masks, and evaluate the
# performance metrics (accuracy, etc.)
from lib.losses import BCELoss, DiceLoss, FocalLoss, BCELoss_TotalVariation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def diceOverlap(y_pred: torch.Tensor, y_true: torch.Tensor, smooth=1e-6):
    """Computes Dice Overlap between predicted and true masks
    Args:
        y_pred (torch.Tensor): Predicted segmentation masks
        y_true (torch.Tensor): Ground truth segmentation masks
        smooth (float): Smoothing factor to avoid division by zero
    Returns:
        float: Dice Overlap score
    """
    # Ensure correct shape
    if y_true.ndim == 3:
        y_true = y_true.unsqueeze(1)
    assert y_pred.shape == y_true.shape, "Shape mismatch between predictions and ground truth"
    # Apply sigmoid to convert logits → probabilities
    y_pred = torch.sigmoid(y_pred)
    # Flatten across spatial dimensions
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    denominator = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
    # Compute Dice Overlap
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return dice.mean().item()

def intersection_over_union(y_pred: torch.Tensor, y_true: torch.Tensor, smooth=1e-6):
    """Computes Intersection over Union (IoU) between predicted and true masks
    Args:
        y_pred (torch.Tensor): Predicted segmentation masks
        y_true (torch.Tensor): Ground truth segmentation masks
        smooth (float): Smoothing factor to avoid division by zero
    Returns:
        float: IoU score
    """
    if y_true.ndim == 3:
        y_true = y_true.unsqueeze(1)
    # Ensure correct shape
    assert y_pred.shape == y_true.shape, "Shape mismatch between predictions and ground truth"
    # Apply sigmoid to convert logits → probabilities
    y_pred = torch.sigmoid(y_pred) 
    # Flatten across spatial dimensions
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) - intersection
    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0.5):
    """
    Computes pixel-wise accuracy for binary segmentation masks.
    
    Args:
        y_pred (torch.Tensor): Model predictions (logits or probabilities), shape [B, 1, H, W]
        y_true (torch.Tensor): Ground truth binary masks, same shape as y_pred
        threshold (float): Threshold for converting probabilities to binary (default 0.5)
    Returns:
        float: Accuracy score (0.0 - 1.0)
    """
    # Ensure correct shape
    if y_true.ndim == 3:
        y_true = y_true.unsqueeze(1)
    assert y_pred.shape == y_true.shape, "Shape mismatch between predictions and ground truth"
    # Match shapes
    if y_true.ndim == 3:
        y_true = y_true.unsqueeze(1)
    if y_pred.ndim == 3:
        y_pred = y_pred.unsqueeze(1)

    # Convert logits → probabilities → binary mask
    y_prob = torch.sigmoid(y_pred)
    y_pred_bin = (y_prob > threshold).float()
    y_true_bin = (y_true > 0.5).float()

    # Flatten to vectors
    y_pred_flat = y_pred_bin.view(-1)
    y_true_flat = y_true_bin.view(-1)

    # Compute counts
    correct = (y_pred_flat == y_true_flat).float().sum()
    total = y_true_flat.numel()

    acc = correct / total
    return acc.item()

def sensitivity(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0.5, smooth=1e-6):
    """
    Computes sensitivity (recall / TPR) for binary segmentation.
    """
    # Ensure same shape
    if y_true.ndim == 3:
        y_true = y_true.unsqueeze(1)
    if y_pred.ndim == 3:
        y_pred = y_pred.unsqueeze(1)
    assert y_pred.shape == y_true.shape, "Shape mismatch between predictions and ground truth"
    # Convert logits → probabilities → binary mask
    y_prob = torch.sigmoid(y_pred)
    y_pred_bin = (y_prob > threshold).float()
    y_true_bin = (y_true > 0.5).float()

    # Flatten
    y_pred_f = y_pred_bin.view(-1)
    y_true_f = y_true_bin.view(-1)

    # True Positives and False Negatives
    TP = (y_pred_f * y_true_f).sum()
    FN = ((1 - y_pred_f) * y_true_f).sum()

    # Sensitivity = TP / (TP + FN)
    sens = (TP + smooth) / (TP + FN + smooth)
    return sens.item()


def specificity(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0.5, smooth=1e-6):
    """
    Computes specificity (TNR) for binary segmentation.
    """
    # Ensure same shape
    if y_true.ndim == 3:
        y_true = y_true.unsqueeze(1)
    if y_pred.ndim == 3:
        y_pred = y_pred.unsqueeze(1)

    assert y_pred.shape == y_true.shape, "Shape mismatch between predictions and ground truth"
    # Convert logits → probabilities → binary mask
    y_prob = torch.sigmoid(y_pred)
    y_pred_bin = (y_prob > threshold).float()
    y_true_bin = (y_true > 0.5).float()

    # Flatten
    y_pred_f = y_pred_bin.view(-1)
    y_true_f = y_true_bin.view(-1)

    # True Negatives and False Positives
    TN = ((1 - y_pred_f) * (1 - y_true_f)).sum()
    FP = (y_pred_f * (1 - y_true_f)).sum()

    # Specificity = TN / (TN + FP)
    spec = (TN + smooth) / (TN + FP + smooth)
    return spec.item()



def evaluate_model(model: nn.Module, data_loader: DataLoader):
    """Evaluates a given model using a test loader

    Args:
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): The data loader for the test set
    Returns:
        String:  a summary of the evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    model.eval()
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # Compute additional metrics (e.g., IoU, Dice, etc.)
            iou = intersection_over_union(outputs, masks)
            dice = diceOverlap(outputs, masks)
            acc = accuracy(masks, outputs)
            sens = sensitivity(masks, outputs)
            spec = specificity(masks, outputs)

            # Log or print the metrics
            return f"IoU: {iou}, Dice: {dice}, Acc: {acc}, Sensitivity: {sens}, Specificity: {spec}"