import torch
import torch.nn as nn
class DiceLoss_fromLogits(nn.Module):
    """
    Dice Loss for binary segmentation.
    Expects raw logits from the model output and binary target masks.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss_fromLogits, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions to (N, H*W) where N is batch size
        # Or simply flatten the entire tensor if it's already (N, 1, H, W)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        # Dice Loss is 1 - Dice Score
        return 1 - dice

class DiceBCE_fromLogits(nn.Module):
    """
    Combined Binary Cross-Entropy (BCE) and Dice Loss for segmentation.
    Both losses are calculated directly from raw model logits for numerical stability.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None, smooth=1e-6):
        super(DiceBCE_fromLogits, self).__init__()

        # You can choose to normalize weights so they sum to 1, or use them as absolute weights.
        # For simplicity, this implementation uses them as absolute weights.
        # If you want them to sum to 1, you could add:
        # total_weight = bce_weight + dice_weight
        # self.bce_weight = bce_weight / total_weight if total_weight > 0 else 0.5
        # self.dice_weight = dice_weight / total_weight if total_weight > 0 else 0.5

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # BCEWithLogitsLoss combines Sigmoid and BCE for numerical stability
        # pos_weight helps with class imbalance (e.g., small foreground objects)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss_fromLogits(smooth=smooth)

    def forward(self, logits, targets):
        # Ensure targets are float (BCEWithLogitsLoss and DiceLoss typically expect float for targets)
        targets = targets.float()

        # Calculate individual losses from logits
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        # Combined loss as a weighted sum
        combined_loss = self.bce_weight * bce + self.dice_weight * dice

        return combined_loss

