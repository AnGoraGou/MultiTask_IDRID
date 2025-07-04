# Modified training loop to integrate routing-based dynamic loss weighting
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from data.classification_dataset import ClassificationDataset
from data.segmentation_dataset import SegmentationDataset,  SegmentationDataset_preAug
from models.model import MultiTaskNet  # Assuming this includes routing
from transforms.joint_transform import JointTransform, JointTransform_preAug, set_seed
from utils.metrics import accuracy, dice_score
from utils.loss import DiceBCE_fromLogits
import matplotlib.pyplot as plt
from datetime import datetime
import os
from itertools import cycle

# ============ CONFIG ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 7e-5
ALPHA = 1.0  # Classification loss weight
BETA = 5.0   # Segmentation loss weight
optimizer_type = "adamw"
WEIGHT_DECAY = 0.03
SPLIT_RATIO = 0.8
set_seed(42)

# Timestamp for unique identification
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = "outputs/models_mtlr"
plot_dir = "outputs/plots_mtlr"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ============ TRANSFORM & DATASET ============
train_transform = JointTransform_preAug(crop_size=2048, resize=(512, 512))
val_transform = JointTransform_preAug(crop_size=2048, resize=(512, 512))
cls_train_transform = JointTransform(crop_size=2048, resize=(512, 512))

cls_dataset = ClassificationDataset(
    image_dir='./../B20Disease20Grading/DiseaseGrading/OriginalImages/TrainingSet',
    label_csv='./../B20Disease20Grading/DiseaseGrading/Groundtruths/IDRiD_Disease_Grading_Training_Labels.csv',
    joint_transform=cls_train_transform
)

#seg_dataset = SegmentationDataset(
#    image_dir='./../A20Segmentation/augmented/images',
#    mask_dirs={ 'OpticDisc': './../A20Segmentation/augmented/masks' },
#    joint_transform=joint_transform
#)

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
cls_train_dataset, cls_val_dataset = random_split(cls_dataset, [int(SPLIT_RATIO*len(cls_dataset)), len(cls_dataset)-int(SPLIT_RATIO*len(cls_dataset))])

seg_train_loader = DataLoader(seg_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
seg_val_loader = DataLoader(seg_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
cls_train_loader = DataLoader(cls_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
cls_val_loader = DataLoader(cls_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============ MODEL, LOSS, OPTIMIZER ============
model = MultiTaskNet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    num_classes=5
    ).to(DEVICE)

criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
pos_weight = torch.tensor([12.0], device=DEVICE)
criterion_seg = DiceBCE_fromLogits(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight).to(DEVICE)

if optimizer_type == "adam":
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif optimizer_type == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif optimizer_type == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
else:
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
best_model_path = os.path.join(model_dir, f"best_mtlr_model_{timestamp}.pth")
final_model_path = os.path.join(model_dir, f"final_mtlr_model_{timestamp}.pth")

# ============ TRAINING LOOP ============
train_losses, val_losses = [], []
cls_accuracies, val_cls_accuracies = [], []
seg_dices, val_seg_dices = [], []
no_improve_epochs, best_val_dice = 0, 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss_epoch, total_cls_acc, total_dice, count = 0, 0, 0, 0
    seg_iter = cycle(seg_train_loader)

    for x_cls, y_cls in cls_train_loader:
        count += 1
        x_cls, y_cls = x_cls.to(DEVICE), y_cls.to(DEVICE)
        x_seg, y_seg = next(seg_iter)
        x_seg, y_seg = x_seg.to(DEVICE), y_seg.to(DEVICE).float()


        optimizer.zero_grad()

        # === Forward pass ===
        out_cls = model(x_cls, task='cls', return_routing=True)
        preds_cls = out_cls['cls_out']
        routing_cls = out_cls['routing_weights']
        loss_cls = criterion_cls(preds_cls, y_cls)
        acc = accuracy(preds_cls.detach(), y_cls)
        total_cls_acc += acc

        out_seg = model(x_seg, task='seg', return_routing=True)
        preds_seg = out_seg['seg_out'].squeeze()
        #print(f'pred_seg  shape:  {preds_seg.shape} max: {preds_seg.max} - min: {preds_seg.min}, y_seg:{y_seg.shape} {y_seg.max}')

        routing_seg = out_seg['routing_weights'].squeeze()
        loss_seg = criterion_seg(preds_seg, y_seg)
        dice = dice_score(preds_seg.detach(), y_seg.detach())
        total_dice += dice

        # === Dynamic Loss Weighting ===
        w_cls = routing_cls[:, 0].mean()
        w_seg = routing_seg[:, 1].mean()
        total_loss = w_cls * ALPHA * loss_cls + w_seg * BETA * loss_seg

        # === Backpropagation ===
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss_epoch += total_loss.item()
        #print("Train Seg Output:", preds_seg.min().item(), preds_seg.max().item())
        #print("Train Seg Target:", y_seg.min().item(), y_seg.max().item())
        
    # === Logging Training Metrics ===
    avg_loss = total_loss_epoch / count
    avg_acc = total_cls_acc / count
    avg_dice = total_dice / count
    train_losses.append(avg_loss)
    cls_accuracies.append(avg_acc)
    seg_dices.append(avg_dice)

    # === VALIDATION ===
    model.eval()
    val_loss_epoch, val_cls_acc, val_dice_total, val_count = 0, 0, 0, 0

    with torch.no_grad():
        for (x_cls_val, y_cls_val), (x_seg_val, y_seg_val) in zip(cls_val_loader, seg_val_loader):
            val_count += 1

            # === Classification ===
            x_cls_val, y_cls_val = x_cls_val.to(DEVICE), y_cls_val.to(DEVICE)
            out_val_cls = model(x_cls_val, task='cls')
            preds_cls_val = out_val_cls['cls_out']
            val_loss_cls = criterion_cls(preds_cls_val, y_cls_val)
            acc_val = accuracy(preds_cls_val, y_cls_val)
            val_cls_acc += acc_val

            # === Segmentation ===
            x_seg_val, y_seg_val = x_seg_val.to(DEVICE), y_seg_val.to(DEVICE)
            out_val_seg = model(x_seg_val, task='seg')
            preds_seg_val = out_val_seg['seg_out'].squeeze()
            val_loss_seg = criterion_seg(preds_seg_val, y_seg_val)
            dice_val = dice_score(preds_seg_val, y_seg_val)
            val_dice_total += dice_val

            val_loss_epoch += (ALPHA * val_loss_cls + BETA * val_loss_seg).item()

    # === Logging Validation Metrics ===
    avg_val_loss = val_loss_epoch / val_count
    avg_val_acc = val_cls_acc / val_count
    avg_val_dice = val_dice_total / val_count

    val_losses.append(avg_val_loss)
    val_cls_accuracies.append(avg_val_acc)
    val_seg_dices.append(avg_val_dice)

    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        torch.save(model.state_dict(), best_model_path)
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= 10:
            print("__ Early stopping triggered.")
            break
            
    print(f"EPOCH [{epoch+1:03}/{NUM_EPOCHS}]")
    print("-" * 60)
    print(f"Train     | Loss: {avg_loss:.4f} | Cls Acc: {avg_acc:.4f} | Seg Dice: {avg_dice:.4f}")
    print(f"Val       | Loss: {avg_val_loss:.4f} | Cls Acc: {avg_val_acc:.4f} | Seg Dice: {avg_val_dice:.4f}")
    print("-" * 60)

# ============ FINAL SAVE ============
torch.save(model.state_dict(), final_model_path)

# ============ PLOT METRICS ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses, label="Val Loss")
ax1.set_title("Loss Over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(cls_accuracies, label="Train Cls Acc")
ax2.plot(val_cls_accuracies, label="Val Cls Acc")
ax2.plot(seg_dices, label="Train Seg Dice")
ax2.plot(val_seg_dices, label="Val Seg Dice")
ax2.set_title("Performance Metrics Over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Metric Value")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plot_path = os.path.join(plot_dir, f"mtlr_metrics_plot_{timestamp}.png")
plt.savefig(plot_path)
plt.close()
