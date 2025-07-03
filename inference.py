import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

from models.model import UnetMultiTaskModel

# --- Configuration ---
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_DIR = '/home/nilgiri/Downloads/archive/A20Segmentation/Segmentation/OriginalImages/TestingSet'
    MASK_DIR = '/home/nilgiri/Downloads/archive/A20Segmentation/Segmentation/AllSegmentationGroundtruths/TestingSet/OpticDisc'
    OUTPUT_DIR = 'test_results_multitask'
    MODEL_PATH = 'outputs/models/final_model_20250703_010502.pth'
    PATCH_SIZE = 2048
    RESIZE_DIM = 512
    ENCODER_NAME = "resnet18"
    NUM_CLASSES = 5
    NUM_SEG_CHANNELS = 1
    NUM_EXPERTS = 4
    DROPOUT_P = 0.3

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# --- Dice Score ---
def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0
    return (2. * intersection) / (union + 1e-8)

# --- Load Model ---
print(f"__ Loading model from: {Config.MODEL_PATH}")
model = UnetMultiTaskModel(
    encoder_name=Config.ENCODER_NAME,
    num_classes=Config.NUM_CLASSES,
    num_seg_channels=Config.NUM_SEG_CHANNELS,
    num_experts=Config.NUM_EXPERTS,
    dropout_p=Config.DROPOUT_P
).to(Config.DEVICE)
model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
model.eval()

# --- Transforms ---
preprocess_transform = transforms.Compose([
    transforms.Resize((Config.RESIZE_DIM, Config.RESIZE_DIM)),
    transforms.ToTensor()
])

# --- Patch Extraction ---
def get_patch_coords(img_shape, patch_size):
    h, w = img_shape[:2]
    x_starts = [0, w - 2 * patch_size, w - patch_size]  # For 4288 → [0, 1240, 2240]
    y_starts = [0, h - patch_size]                      # For 2848 → [0, 800]
    coords = [(x, y) for y in y_starts for x in x_starts]
    return coords

# --- Inference Function ---
def run_inference():
    image_paths = sorted(glob.glob(os.path.join(Config.IMG_DIR, '*.jpg')))
    mask_paths = sorted(glob.glob(os.path.join(Config.MASK_DIR, '*.tif')))

    if not image_paths or not mask_paths:
        print("Error: No images or masks found in the specified directories.")
        return

    dice_scores = []

    print(f"Total images for inference: {len(image_paths)}")

    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Processing images"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        gt_mask = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)

        # Blank prediction mask
        stitched_pred = np.zeros((h, w), dtype=np.uint8)

        patch_preds = []
        patch_coords = get_patch_coords((h, w), Config.PATCH_SIZE)

        for (x, y) in patch_coords:
            patch_np = img_np[y:y + Config.PATCH_SIZE, x:x + Config.PATCH_SIZE]
            patch_img = Image.fromarray(patch_np)
            patch_tensor = preprocess_transform(patch_img).unsqueeze(0).to(Config.DEVICE)

            # --- Segmentation Inference ---
            with torch.no_grad():
                seg_output = model(patch_tensor, task="segmentation")
                pred_mask = torch.sigmoid(seg_output).squeeze().cpu().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

            # Resize to original patch size (from 512 back to 2048)
            pred_resized = np.array(Image.fromarray(pred_mask).resize((Config.PATCH_SIZE, Config.PATCH_SIZE), Image.NEAREST))
            stitched_pred[y:y + Config.PATCH_SIZE, x:x + Config.PATCH_SIZE] = pred_resized

            # --- Classification Inference ---
            with torch.no_grad():
                logits = model(patch_tensor, task="classification")
                pred_class = torch.argmax(logits, dim=1).item()
                patch_preds.append(pred_class)

        # --- Dice Score ---
        dice = dice_score((stitched_pred > 127).astype(np.uint8), gt_mask)
        dice_scores.append(dice)

        # --- Save Segmentation Visualization ---
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(img_np)
        axs[0].set_title("Original Image")
        axs[1].imshow(gt_mask, cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[2].imshow(stitched_pred, cmap='gray')
        axs[2].set_title(f"Prediction - Dice: {dice:.3f}")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, f"{base_name}_seg.png"))
        plt.close()

        # --- Save Classification Result ---
        final_class = max(set(patch_preds), key=patch_preds.count) if patch_preds else "N/A"
        with open(os.path.join(Config.OUTPUT_DIR, f"{base_name}_class.txt"), 'w') as f:
            f.write(f"Patch Votes: {patch_preds}\n")
            f.write(f"Final Predicted Class (Majority Voting): {final_class}\n")

    # --- Summary ---
    if dice_scores:
        avg_dice = np.mean(dice_scores)
        print(f"\n__ Average Dice over Test Set: {avg_dice:.4f}")
    else:
        print("\nNo Dice scores calculated.")

if __name__ == "__main__":
    run_inference()
