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



class JointTransform:
    def __init__(self, mode='train', crop_size=1024, resize=(512, 512), hflip=True, vflip=True, color_jitter=True, rotation=True):
        assert mode in ['train', 'val'], "Mode must be 'train' or 'val'"
        self.mode = mode
        self.crop_size = crop_size
        self.resize = resize
        self.hflip = hflip
        self.vflip = vflip
        self.color_jitter = color_jitter
        self.rotation = rotation

        self.color_jitter_transform = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
        )

    def __call__(self, image, masks):
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        masks = [TF.crop(mask, i, j, h, w) for mask in masks]
        
        if self.mode == 'train':
            # Random crop
            #i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            #image = TF.crop(image, i, j, h, w)
            #masks = [TF.crop(mask, i, j, h, w) for mask in masks]

            if self.hflip and random.random() > 0.5:
                image = TF.hflip(image)
                masks = [TF.hflip(mask) for mask in masks]

            if self.vflip and random.random() > 0.5:
                image = TF.vflip(image)
                masks = [TF.vflip(mask) for mask in masks]

            if self.rotation:
                angle = random.choice([0, 90, 180, 270])
                image = TF.rotate(image, angle)
                masks = [TF.rotate(mask, angle) for mask in masks]

            if self.color_jitter:
                image = self.color_jitter_transform(image)

        # Resize for both train and val
        image = TF.resize(image, self.resize)
        masks = [TF.resize(mask, self.resize) for mask in masks]

        # To Tensor and binarize mask
        image = TF.to_tensor(image)
        masks = [(TF.to_tensor(mask).squeeze(0) > 0.5).float() for mask in masks]

        return image, masks

        
        
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dirs, joint_transform=None):
        self.image_dir = image_dir
        self.mask_dirs = mask_dirs
        self.joint_transform = joint_transform

        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.suffix_map = {
            'OpticDisc': '_OD.tif'
        }

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        base_name = os.path.splitext(image_name)[0]
        img_path = os.path.join(self.image_dir, image_name)
        image = Image.open(img_path).convert('RGB')

        masks = []
        for lesion, mask_dir in self.mask_dirs.items():
            suffix = self.suffix_map.get(lesion, '.png')
            mask_path = os.path.join(mask_dir, base_name + suffix)

            mask = Image.open(mask_path).convert('L') if os.path.exists(mask_path) else Image.new('L', image.size)
            masks.append(mask)

        if self.joint_transform:
            image, masks = self.joint_transform(image, masks)

        masks_tensor = torch.stack(masks, dim=0)  # [1, H, W]
        return image, masks_tensor.squeeze(0)     # [3, H, W], [H, W]




class DiceLoss_fromLogits(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss_fromLogits, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()

class DiceBCE_fromLogits(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None, smooth=1e-6):
        super(DiceBCE_fromLogits, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss_fromLogits(smooth=smooth)

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


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
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.003
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



# Define transforms
train_transform = JointTransform(mode='train', crop_size=2048, resize=(512, 512))
val_transform = JointTransform(mode='val', resize=(512, 512))


# Train Dataset
train_dataset = SegmentationDataset(
    image_dir='./../A20Segmentation/augmented/train/images',
    mask_dirs={'OpticDisc': './../A20Segmentation/augmented/train/masks'},
    joint_transform=train_transform  # Apply augmentations only on training set
)

# Validation Dataset (no augmentations)
val_dataset = SegmentationDataset(
    image_dir='./../A20Segmentation/augmented/val/images',
    mask_dirs={'OpticDisc': './../A20Segmentation/augmented/val/masks'},
    joint_transform=val_transform  # No augmentation for validation
)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)


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

