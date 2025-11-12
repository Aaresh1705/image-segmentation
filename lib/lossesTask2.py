import torch
import torch.nn as nn
import torch.nn.functional as F

class WeakPointBCELoss(nn.Module):
    """
    Weakly-supervised BCE loss using sparse (x, y, label) points.
    Args:
        tv_weight (float): optional weight for smoothness regularization
    """
    def __init__(self, tv_weight=0.0):
        super().__init__()
        self.tv_weight = tv_weight
        self.name = 'WeakPointBCELoss'

    def forward(self, y_pred, points_batch):
        """
        Args:
            y_pred: Tensor (B, 1, H, W) — model output logits
            points_batch: list of [N_i x 3] tensors (x, y, lesion)
        Returns:
            Scalar loss
        """
        B, _, H, W = y_pred.shape
        y_prob = torch.sigmoid(y_pred)

        total_loss = 0.0
        count = 0

        for b in range(B):
            points = points_batch[b]  # (N, 3)
            if len(points) == 0:
                continue

            xs = points[:, 0].long().clamp(0, W - 1)
            ys = points[:, 1].long().clamp(0, H - 1)
            labels = points[:, 2]

            # Sample model predictions at annotated points
            preds = y_prob[b, 0, ys, xs]

            # BCE for sparse labels
            loss = F.binary_cross_entropy(preds, labels, reduction='mean')
            total_loss += loss
            count += 1

        if count > 0:
            total_loss = total_loss / count

        # Optional smoothness term
        if self.tv_weight > 0:
            diff_h = torch.abs(y_prob[:, :, :, 1:] - y_prob[:, :, :, :-1])
            diff_v = torch.abs(y_prob[:, :, 1:, :] - y_prob[:, :, :-1, :])
            tv_loss = (torch.sum(diff_h) + torch.sum(diff_v)) / B
            total_loss += self.tv_weight * tv_loss

        return total_loss
