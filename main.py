import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from data.classification_dataset import ClassificationDataset
from data.segmentation_dataset import SegmentationDataset
from models.modular_multitask_model import ModularMultiTaskModel
from transforms.joint_transform import JointTransform
from utils.metrics import accuracy, dice_score
import matplotlib.pyplot as plt
import random

# ============ CONFIG ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 7e-5
ALPHA = 1.0  # Classification loss weight
BETA = 6.0   # Segmentation loss weight

# ============ TRANSFORM ============
joint_transform = JointTransform(crop_size=2048, resize=(512, 512))

# ============ DATASET ============
cls_dataset = ClassificationDataset(
    image_dir='./../B20Disease20Grading/DiseaseGrading/OriginalImages/TrainingSet',
    label_csv='./../B20Disease20Grading/DiseaseGrading/Groundtruths/IDRiD_Disease_Grading_Training_Labels.csv',
    joint_transform=joint_transform
)

seg_dataset = SegmentationDataset(
    image_dir='./../A20Segmentation/Segmentation/OriginalImages/TrainingSet',
    mask_dirs={
        'OpticDisc': './../A20Segmentation/Segmentation/AllSegmentationGroundtruths/TrainingSet/Optic_Disc',
    },
    joint_transform=joint_transform
)


#calculate lengths for the 80:20 split
seg_total_size = len(seg_dataset)
seg_train_size = int(0.8 * seg_total_size)
seg_val_size = seg_total_size - seg_train_size # Ensure all samples are used

# Perform the random split
seg_train_dataset, seg_val_dataset = random_split(seg_dataset, [seg_train_size, seg_val_size])

print(f"Segmentation Dataset - Total: {seg_total_size}, Train: {len(seg_train_dataset)}, Validation: {len(seg_val_dataset)}")

# Create DataLoaders
seg_train_loader = DataLoader(seg_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
seg_val_loader = DataLoader(seg_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Calculate lengths for the 80:20 split
cls_total_size = len(cls_dataset)
cls_train_size = int(0.8 * cls_total_size)
cls_val_size = cls_total_size - cls_train_size # Ensure all samples are used

# Perform the random split
cls_train_dataset, cls_val_dataset = random_split(cls_dataset, [cls_train_size, cls_val_size])

print(f"Classification Dataset - Total: {cls_total_size}, Train: {len(cls_train_dataset)}, Validation: {len(cls_val_dataset)}")

# Create DataLoaders

cls_train_loader = DataLoader(cls_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
cls_val_loader = DataLoader(cls_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#cls_loader = DataLoader(cls_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
#seg_loader = DataLoader(seg_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)i

exit()


# ============ MODEL, LOSS, OPTIMIZER ============
model = ModularMultiTaskModel(num_experts=4, num_classes=5, num_seg_channels=1).to(DEVICE)
criterion_cls = nn.CrossEntropyLoss()
criterion_seg = nn.BCEWithLogitsLoss()

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

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Instantiate the CosineAnnealingLR scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

combined_loss_fn = DiceBCE_fromLogits().to(DEVICE)

# ============ TRAINING ============
train_losses = []
cls_accuracies = []
seg_dices = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss_epoch, total_cls_acc, total_dice, count = 0, 0, 0, 0

    cls_iter = iter(cls_loader)
    seg_iter = iter(seg_loader)
    num_batches = min(len(cls_loader), len(seg_loader))

    for _ in range(num_batches):
        count += 1
        optimizer.zero_grad()

        # === Classification ===
        try:
            x_cls, y_cls = next(cls_iter)
            x_cls, y_cls = x_cls.to(DEVICE), y_cls.to(DEVICE)
            preds_cls = model(x_cls, task="classification")
            loss_cls = criterion_cls(preds_cls, y_cls)
            acc = accuracy(preds_cls.detach(), y_cls)
            total_cls_acc += acc
        except StopIteration:
            loss_cls = 0.0

        # === Segmentation ===
        try:
            x_seg, y_seg = next(seg_iter)
            x_seg, y_seg = x_seg.to(DEVICE), y_seg.to(DEVICE)
            preds_seg = model(x_seg, task="segmentation")
            loss_seg = combined_loss_fn(preds_seg, y_seg) # criterion_seg(preds_seg, y_seg)
            dice = dice_score(preds_seg.detach(), y_seg)
            total_dice += dice
        except StopIteration:
            loss_seg = 0.0

        # === Total Loss ===
        if isinstance(loss_cls, torch.Tensor) and isinstance(loss_seg, torch.Tensor):
            total_loss = ALPHA * loss_cls + BETA * loss_seg
        elif isinstance(loss_cls, torch.Tensor):
            total_loss = loss_cls
        elif isinstance(loss_seg, torch.Tensor):
            total_loss = loss_seg
        else:
            continue

        total_loss.backward()
        optimizer.step()
        total_loss_epoch += total_loss.item()
        scheduler.step()

    # === Epoch Metrics ===
    avg_loss = total_loss_epoch / count
    avg_acc = total_cls_acc / count
    avg_dice = total_dice / count
    train_losses.append(avg_loss)
    cls_accuracies.append(avg_acc)
    seg_dices.append(avg_dice)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Loss: {avg_loss:.4f} | "
          f"Cls Acc: {avg_acc:.4f} | "
          f"Seg Dice: {avg_dice:.4f}")

# ============ SAVE MODEL ============
torch.save(model.state_dict(), "multitask_model_BCEDice.pth")

# ============ PLOT METRICS ============
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Total Loss")
plt.plot(cls_accuracies, label="Classification Accuracy")
plt.plot(seg_dices, label="Segmentation Dice")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Training Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_metrics_plot_BCE_dice.png")
plt.close()

