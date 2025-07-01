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
from utils.loss import DiceBCE_fromLogits
import matplotlib.pyplot as plt
import random
from datetime import datetime
import os

# ============ CONFIG ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
ALPHA = 1.0  # Classification loss weight
BETA = 5.0   # Segmentation loss weight
optimizer_type = "adamw"  # choose from: "adam", "adamw", "sgd"
WEIGHT_DECAY = 0.07

# Timestamp for unique identification
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output directories
model_dir = "outputs/models"
plot_dir = "outputs/plots"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

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
#seg_loader = DataLoader(seg_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# ============ MODEL, LOSS, OPTIMIZER ============
model = ModularMultiTaskModel(num_experts=4, num_classes=5, num_seg_channels=1).to(DEVICE)
criterion_cls = nn.CrossEntropyLoss()
criterion_seg = nn.BCEWithLogitsLoss()


# Save best model based on validation Dice
best_model_path = os.path.join(model_dir, f"best_model_{timestamp}.pth")
# Save final model
final_model_path = os.path.join(model_dir, f"final_model_{timestamp}.pth")


#optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

if optimizer_type == "adam":
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif optimizer_type == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif optimizer_type == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
else:
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

# Instantiate the CosineAnnealingLR scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

combined_loss_fn = DiceBCE_fromLogits().to(DEVICE)

# ============ TRAINING ============

#initialize tracking lists
train_losses, val_losses = [], []
cls_accuracies, val_cls_accuracies = [], []
seg_dices, val_seg_dices = [], []

best_val_dice = 0.0  # For saving best model

for epoch in range(NUM_EPOCHS):
    # ========== TRAIN ==========
    model.train()
    total_loss_epoch, total_cls_acc, total_dice, count = 0, 0, 0, 0

    cls_iter = iter(cls_train_loader)
    seg_iter = iter(seg_train_loader)
    num_batches = min(len(cls_train_loader), len(seg_train_loader))

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
            loss_seg = combined_loss_fn(preds_seg, y_seg)
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

    # Training metrics
    avg_loss = total_loss_epoch / count
    avg_acc = total_cls_acc / count
    avg_dice = total_dice / count
    train_losses.append(avg_loss)
    cls_accuracies.append(avg_acc)
    seg_dices.append(avg_dice)

    # ========== VALIDATION ==========
    model.eval()
    val_loss_epoch, val_cls_acc, val_dice_total, val_count = 0, 0, 0, 0

    with torch.no_grad():
        for (x_cls_val, y_cls_val), (x_seg_val, y_seg_val) in zip(cls_val_loader, seg_val_loader):
            val_count += 1

            # === Classification ===
            x_cls_val, y_cls_val = x_cls_val.to(DEVICE), y_cls_val.to(DEVICE)
            preds_cls_val = model(x_cls_val, task="classification")
            val_loss_cls = criterion_cls(preds_cls_val, y_cls_val)
            acc_val = accuracy(preds_cls_val, y_cls_val)
            val_cls_acc += acc_val

            # === Segmentation ===
            x_seg_val, y_seg_val = x_seg_val.to(DEVICE), y_seg_val.to(DEVICE)
            preds_seg_val = model(x_seg_val, task="segmentation")
            val_loss_seg = combined_loss_fn(preds_seg_val, y_seg_val)
            dice_val = dice_score(preds_seg_val, y_seg_val)
            val_dice_total += dice_val

            val_loss_epoch += (ALPHA * val_loss_cls + BETA * val_loss_seg).item()

    avg_val_loss = val_loss_epoch / val_count
    avg_val_acc = val_cls_acc / val_count
    avg_val_dice = val_dice_total / val_count

    val_losses.append(avg_val_loss)
    val_cls_accuracies.append(avg_val_acc)
    val_seg_dices.append(avg_val_dice)

    # Save best model based on validation Dice
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        torch.save(model.state_dict(), best_model_path)
        #torch.save(model.state_dict(), "best_multitask_model_validated.pth")

    #print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
    #      f"Train Loss: {avg_loss:.4f} | Cls Acc: {avg_acc:.4f} | Seg Dice: {avg_dice:.4f} || "
    #      f"Val Loss: {avg_val_loss:.4f} | Val Cls Acc: {avg_val_acc:.4f} | Val Seg Dice: {avg_val_dice:.4f}")
    
    print(f"\nEpoch [{epoch+1:03}/{NUM_EPOCHS}]")
    print("-" * 60)
    print(f"Train     | Loss: {avg_loss:.4f} | Cls Acc: {avg_acc:.4f} | Seg Dice: {avg_dice:.4f}")
    print(f"Val       | Loss: {avg_val_loss:.4f} | Cls Acc: {avg_val_acc:.4f} | Seg Dice: {avg_val_dice:.4f}")
    print("-" * 60)


# ============ FINAL SAVE ============
#torch.save(model.state_dict(), "final_multitask_model_validated.pth")
torch.save(model.state_dict(), final_model_path)

# ============ PLOT METRICS ============

# Create a 1x2 grid of subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Subplot 1: Losses ---
ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses, label="Val Loss")
ax1.set_title("Loss Over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

# --- Subplot 2: Accuracy & Dice ---
ax2.plot(cls_accuracies, label="Train Cls Acc")
ax2.plot(val_cls_accuracies, label="Val Cls Acc")
ax2.plot(seg_dices, label="Train Seg Dice")
ax2.plot(val_seg_dices, label="Val Seg Dice")
ax2.set_title("Performance Metrics Over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Metric Value")
ax2.legend()
ax2.grid(True)

# Final layout
plt.tight_layout()

# Save with timestamp
plot_path = os.path.join(plot_dir, f"metrics_plot_{timestamp}.png")
plt.savefig(plot_path)
plt.close()


'''
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.plot(cls_accuracies, label="Train Cls Acc")
plt.plot(val_cls_accuracies, label="Val Cls Acc")
plt.plot(seg_dices, label="Train Seg Dice")
plt.plot(val_seg_dices, label="Val Seg Dice")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Training & Validation Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(plot_dir, f"metrics_plot_{timestamp}.png")
plt.savefig(plot_path)
plt.close()
'''
