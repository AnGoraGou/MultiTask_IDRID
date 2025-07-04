# baseline_classification.py: Train classification baseline separately

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from datetime import datetime
from utils.metrics import accuracy
from utils.loss import GeneralizedCELoss
from transforms.joint_transform import JointTransform, set_seed
from data.classification_dataset import ClassificationDataset
from torchvision import models

# ============ CONFIG ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.003
SEED = 42
set_seed(SEED)

# Output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = "outputs/cls_baseline_models"
plot_dir = "outputs/cls_baseline_plots"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ============ TRANSFORM & DATASET ============
joint_transform = JointTransform(crop_size=2048, resize=(512, 512))

cls_dataset = ClassificationDataset(
    image_dir='./../B20Disease20Grading/DiseaseGrading/OriginalImages/TrainingSet',
    label_csv='./../B20Disease20Grading/DiseaseGrading/Groundtruths/IDRiD_Disease_Grading_Training_Labels.csv',
    joint_transform=joint_transform
)
cls_train, cls_val = random_split(cls_dataset, [int(0.7 * len(cls_dataset)), len(cls_dataset) - int(0.7 * len(cls_dataset))])
cls_train_loader = DataLoader(cls_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
cls_val_loader = DataLoader(cls_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============ MODEL, LOSS, OPTIMIZER ============
cls_model = models.resnet18(pretrained=True)
cls_model.fc = nn.Linear(cls_model.fc.in_features, 5)
cls_model = cls_model.to(DEVICE)
cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
cls_optimizer = optim.AdamW(cls_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
cls_scheduler = CosineAnnealingLR(cls_optimizer, T_max=NUM_EPOCHS)

cls_train_losses, cls_val_losses = [], []
cls_train_accs, cls_val_accs = [], []
cls_best_acc = 0.0

# ============ TRAIN LOOP ============
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    # --- Train ---
    cls_model.train()
    cls_loss_sum, cls_acc_sum = 0.0, 0.0
    for imgs, labels in cls_train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        cls_optimizer.zero_grad()
        outputs = cls_model(imgs)
        loss = cls_criterion(outputs, labels)
        loss.backward()
        cls_optimizer.step()
        cls_loss_sum += loss.item()
        cls_acc_sum += accuracy(outputs, labels)
    cls_scheduler.step()
    cls_train_losses.append(cls_loss_sum / len(cls_train_loader))
    cls_train_accs.append(cls_acc_sum / len(cls_train_loader))

    # --- Validation ---
    cls_model.eval()
    val_cls_loss, val_cls_acc = 0.0, 0.0
    with torch.no_grad():
        for imgs, labels in cls_val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = cls_model(imgs)
            loss = cls_criterion(outputs, labels)
            val_cls_loss += loss.item()
            val_cls_acc += accuracy(outputs, labels)
    cls_val_losses.append(val_cls_loss / len(cls_val_loader))
    cls_val_accs.append(val_cls_acc / len(cls_val_loader))
    if cls_val_accs[-1] > cls_best_acc:
        cls_best_acc = cls_val_accs[-1]
        torch.save(cls_model.state_dict(), os.path.join(model_dir, f"cls_best_model_{timestamp}.pth"))

    # Log epoch summary
    print(f"Train Loss: {cls_train_losses[-1]:.4f}, Val Loss: {cls_val_losses[-1]:.4f}, Val Acc: {cls_val_accs[-1]:.4f}")

# Save final model
torch.save(cls_model.state_dict(), os.path.join(model_dir, f"cls_final_model_{timestamp}.pth"))

# ============ PLOTS ============
plt.figure(figsize=(8, 6))
plt.plot(cls_train_losses, label='Train Loss')
plt.plot(cls_val_losses, label='Val Loss')
plt.plot(cls_val_accs, label='Val Accuracy')
plt.title("Classification Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"cls_training_{timestamp}.png"))
plt.close()
