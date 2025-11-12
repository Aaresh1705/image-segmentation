from .losses import BCELoss, FocalLoss, BCELoss_TotalVariation, BCELoss_PositiveWeights

all_losses = [BCELoss_PositiveWeights,BCELoss, FocalLoss, BCELoss_TotalVariation]