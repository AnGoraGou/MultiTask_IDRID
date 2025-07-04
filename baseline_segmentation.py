import os
import random
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from transforms.joint_transform import set_seed  # Make sure this function exists


# Dice metric
def dice_score(preds, targets, threshold=0.5, eps=1e-7):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    if preds.shape != targets.shape:
        preds = preds.squeeze(1)
    intersection = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item()

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.03
SEED = 42
set_seed(SEED)

# Output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = "outputs/seg_baseline_models"
plot_dir = "outputs/seg_baseline_plots"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Dataset

'''
dataset = SegmentationDataset(
    image_dir='./../A20Segmentation/augmented/images',
    mask_dirs={'OpticDisc': './../A20Segmentation/augmented/masks'},
    joint_transform=joint_transform
)
train_set, val_set = random_split(dataset, [int(0.7 * len(dataset)), len(dataset) - int(0.7 * len(dataset))])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
'''
# ============ TRANSFORM & DATASET ============
train_transform = JointTransform_preAug(crop_size=2048, resize=(512, 512))
val_transform = JointTransform_preAug(crop_size=2048, resize=(512, 512))


# Train Dataset
seg_train_dataset = SegmentationDataset_preAug(
    image_dir='./../A20Segmentation/augmented/train/images',
    mask_dirs={'OpticDisc': './../A20Segmentation/augmented/train/masks'},
    joint_transform=train_transform  # Apply augmentations only on training set
)

# Validation Dataset (no augmentations)
seg_val_dataset = SegmentationDataset_preAug(
    image_dir='./../A20Segmentation/augmented/val/images',
    mask_dirs={'OpticDisc': './../A20Segmentation/augmented/val/masks'},
    joint_transform=val_transform  # No augmentation for validation
)

# Splits and loaders
#seg_train_dataset, seg_val_dataset = random_split(seg_dataset, [int(SPLIT_RATIO*len(seg_dataset)), len(seg_dataset)-int(SPLIT_RATIO*len(seg_dataset))])


train_loader = DataLoader(seg_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(seg_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)



# Model
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)

pos_weight = torch.tensor([12.0], device=DEVICE)
criterion = DiceBCE_fromLogits(bce_weight=0.6, dice_weight=0.4, pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Training Loop
best_dice = 0.0
train_losses, val_losses, train_dices, val_dices = [], [], [], []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    model.train()
    running_loss, running_dice = 0.0, 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE).float()
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds.squeeze(), masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_dice += dice_score(preds.squeeze(), masks)
    scheduler.step()
    train_losses.append(running_loss / len(train_loader))
    train_dices.append(running_dice / len(train_loader))

    model.eval()
    val_loss, val_dice = 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE).float()
            preds = model(imgs)
            val_loss += criterion(preds.squeeze(), masks).item()
            val_dice += dice_score(preds.squeeze(), masks)
    val_losses.append(val_loss / len(val_loader))
    val_dices.append(val_dice / len(val_loader))

    if val_dices[-1] > best_dice:
        best_dice = val_dices[-1]
        torch.save(model.state_dict(), os.path.join(model_dir, f"seg_best_model_{timestamp}.pth"))

    print(f"Train Loss: {train_losses[-1]:.4f} || Train Dice: {train_dices[-1]:.4f} \nVal Loss: {val_losses[-1]:.4f} ||  Val Dice: {val_dices[-1]:.4f}")

# Save final model
torch.save(model.state_dict(), os.path.join(model_dir, f"seg_final_model_{timestamp}.pth"))

# Plot
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.plot(val_dices, label='Val Dice')
plt.title("Segmentation Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"seg_training_{timestamp}.png"))
plt.close()

