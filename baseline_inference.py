#baseline_inference.py: Inference for classification and segmentation baselines
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn.functional as F
from torchvision import transforms, models
import segmentation_models_pytorch as smp

#from utils.metrics import dice_score

def dice_score(preds, targets, eps=1e-7):
    # Assumes preds and targets are binary tensors of same shape
    preds = preds.astype(float)
    targets = targets.astype(float)
    #print(f'Max: {preds.max()} || {targets.max()}')

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + eps) / (union + eps)
    #print(f'Dice: {dice:.4f}')
    return dice




# --- Configuration ---
class Config:
    RUN_SEGMENTATION = True
    RUN_CLASSIFICATION = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths for Segmentation Task
    SEG_IMG_DIR = '/home/nilgiri/Downloads/archive/A20Segmentation/Segmentation/OriginalImages/TestingSet'
    SEG_MASK_DIR = '/home/nilgiri/Downloads/archive/A20Segmentation/Segmentation/AllSegmentationGroundtruths/TestingSet/OpticDisc'
    SEG_MODEL_DIR = '/home/nilgiri/Downloads/archive/idrid_project/outputs/models' #'outputs/seg_baseline_models'

    # Paths for Classification Task
    CLASSIF_IMG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'B20Disease20Grading', 'DiseaseGrading', 'OriginalImages', 'TestingSet'))
    CLASSIF_GT_LABELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'B20Disease20Grading', 'DiseaseGrading', 'Groundtruths', 'IDRiD_Disease_Grading_Testing_Labels.csv'))
    CLS_MODEL_DIR = '/home/nilgiri/Downloads/archive/idrid_project/outputs/models' #'outputs/baseline_models'

    OUTPUT_BASE_DIR = 'test_results'
    OUTPUT_SUBDIR = 'baseline_separate_models'
    OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, OUTPUT_SUBDIR)

    # Image processing dimensions
    ORIGINAL_CROP_SIZE = 2048
    MODEL_INPUT_SIZE = 512

    # Model parameters
    SEG_ENCODER_NAME = "resnet18"
    SEG_ENCODER_WEIGHTS = "imagenet"
    SEG_OUT_CHANNELS = 1

    CLS_NUM_CLASSES = 5


# ============ CONFIG ============
RUN_SEGMENTATION = True
RUN_CLASSIFICATION = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = '/home/nilgiri/Downloads/archive/A20Segmentation/Segmentation/OriginalImages/TestingSet'
MASK_DIR = '/home/nilgiri/Downloads/archive/A20Segmentation/Segmentation/AllSegmentationGroundtruths/TestingSet/OpticDisc'
OUTPUT_DIR = 'test_results/baseline'
CLS_BASELINE_DIR = 'outputs/baseline_models'
SEG_BASELINE_DIR = 'outputs/seg_baseline_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# --- Model Loading ---
seg_model = None
cls_model = None

if Config.RUN_SEGMENTATION:
    try:
        seg_model_path = sorted(glob.glob(os.path.join(Config.SEG_MODEL_DIR, 'seg_best_model_*.pth')))[-1]
        print(f"Loading segmentation model from {seg_model_path}")
        seg_model = smp.Unet(
            encoder_name=Config.SEG_ENCODER_NAME,
            encoder_weights=Config.SEG_ENCODER_WEIGHTS,
            in_channels=3,
            classes=Config.SEG_OUT_CHANNELS
        ).to(Config.DEVICE)
        seg_model.load_state_dict(torch.load(seg_model_path, map_location=Config.DEVICE))
        seg_model.eval()
    except IndexError:
        print(f"Error: No segmentation models found in {Config.SEG_MODEL_DIR}. Set RUN_SEGMENTATION = False or check path.")
        Config.RUN_SEGMENTATION = False
    except Exception as e:
        print(f"Error loading segmentation model: {e}")
        Config.RUN_SEGMENTATION = False

if Config.RUN_CLASSIFICATION:
    try:
        cls_model_path = sorted(glob.glob(os.path.join(Config.CLS_MODEL_DIR, 'cls_best_model_*.pth')))[-1]
        print(f"Loading classification model from {cls_model_path}")
        cls_model = models.resnet18(pretrained=False)
        cls_model.fc = torch.nn.Linear(cls_model.fc.in_features, Config.CLS_NUM_CLASSES)
        cls_model = cls_model.to(Config.DEVICE)
        cls_model.load_state_dict(torch.load(cls_model_path, map_location=Config.DEVICE))
        cls_model.eval()
    except IndexError:
        print(f"Error: No classification models found in {Config.CLS_MODEL_DIR}. Set RUN_CLASSIFICATION = False or check path.")
        Config.RUN_CLASSIFICATION = False
    except Exception as e:
        print(f"Error loading classification model: {e}")
        Config.RUN_CLASSIFICATION = False


# --- Image Transformations ---
preprocess_transform = transforms.Compose([
    transforms.Resize((Config.MODEL_INPUT_SIZE, Config.MODEL_INPUT_SIZE)),
    transforms.ToTensor()
])

# ============ REVISED FUNCTIONS FOR GRID PATCHING ============

def get_patches_and_coords(img_np: np.ndarray, patch_size: int = Config.ORIGINAL_CROP_SIZE) -> list:
    """
    Extracts non-overlapping patches from an image, including handling edges.
    Pads the image if necessary to ensure all patches are of patch_size.

    Args:
        img_np: The input image as a NumPy array (H, W, C).
        patch_size: The desired size of each square patch.

    Returns:
        A list of tuples: (patch_np, (y_start, x_start, y_end, x_end_original), (y_padding, x_padding)).
        y_end, x_end_original are coords in the *original* image.
        y_padding, x_padding are the amounts of padding applied to *this specific patch*.
    """
    h, w = img_np.shape[:2]
    patches = []
    
    # Pad the image to ensure dimensions are multiples of patch_size
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    
    # You might want to pad with zeros or reflections depending on your training
    # For now, zero padding is simple. Pad symmetrically if possible.
    padded_img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    
    # Get new dimensions
    padded_h, padded_w = padded_img_np.shape[:2]

    for y in range(0, padded_h, patch_size):
        for x in range(0, padded_w, patch_size):
            patch = padded_img_np[y : y + patch_size, x : x + patch_size]
            
            # Calculate original coordinates for stitching
            original_y_end = min(y + patch_size, h)
            original_x_end = min(x + patch_size, w)
            
            # Calculate padding for this specific patch
            # This is tricky because the patch itself isn't padded, the whole image is.
            # We need to know how much of *this patch* corresponds to the original image.
            # Simplest approach: stitch into a padded canvas, then crop final result.
            
            patches.append((patch, (y, x))) # Store patch and its top-left coordinate in the padded image

    return patches, padded_h, padded_w


def stitch_all_patches_back(predicted_patches: list, original_h: int, original_w: int, 
                             padded_h: int, padded_w: int,
                             model_output_size: int = Config.MODEL_INPUT_SIZE,
                             patch_size_original: int = Config.ORIGINAL_CROP_SIZE) -> np.ndarray:
    """
    Stitches predicted patches back into an image of original dimensions.
    Assumes predicted_patches are already resized from model_output_size to patch_size_original.

    Args:
        predicted_patches: A list of tuples (resized_patch_np, (y_start, x_start))
                         where y_start, x_start are coordinates in the *padded* image.
        original_h: Original height of the image (before padding).
        original_w: Original width of the image (before padding).
        padded_h: Height of the image after padding.
        padded_w: Width of the image after padding.
        model_output_size: Size of the model's direct output (e.g., 512x512).
        patch_size_original: The size of the original patch (e.g., 2048x2048) before resizing for model input.

    Returns:
        The stitched image as a NumPy array, cropped to original_h, original_w.
    """
    # Create a canvas of the padded size
    stitched_padded = np.zeros((padded_h, padded_w), dtype=np.uint8)

    for pred_patch_512, (y_start, x_start) in predicted_patches:
        # Resize the 512x512 prediction back to 2048x2048 for stitching
        resized_pred_patch = np.array(Image.fromarray(pred_patch_512).resize((patch_size_original, patch_size_original), Image.NEAREST))
        
        stitched_padded[y_start : y_start + patch_size_original, x_start : x_start + patch_size_original] = resized_pred_patch

    # Crop the stitched result back to the original image dimensions
    final_stitched = stitched_padded[0:original_h, 0:original_w]
    return final_stitched


# ============ INFERENCE ============
def run_inference():
    # --- Load Classification Ground Truths ---
    classification_ground_truths = {}
    if Config.RUN_CLASSIFICATION:
        try:
            df_labels = pd.read_csv(Config.CLASSIF_GT_LABELS_PATH)
            if 'Image Name' in df_labels.columns and 'Retinopathy grade' in df_labels.columns:
                for index, row in df_labels.iterrows():
                    image_base_name = os.path.splitext(row['Image Name'])[0]
                    classification_ground_truths[image_base_name] = int(row['Retinopathy grade'])
                print(f"Loaded {len(classification_ground_truths)} classification ground truth labels.")
            else:
                print(f"Error: CSV file '{Config.CLASSIF_GT_LABELS_PATH}' must contain 'Image Name' and 'Retinopathy grade' columns.")
                Config.RUN_CLASSIFICATION = False
        except FileNotFoundError:
            print(f"Error: Classification ground truth file not found at {Config.CLASSIF_GT_LABELS_PATH}")
            Config.RUN_CLASSIFICATION = False
        except Exception as e:
            print(f"Error loading classification ground truth CSV: {e}")
            Config.RUN_CLASSIFICATION = False

    image_paths_for_processing = sorted(glob.glob(os.path.join(Config.SEG_IMG_DIR, '*.jpg')))
    mask_paths_for_processing = sorted(glob.glob(os.path.join(Config.SEG_MASK_DIR, '*.png')))
    print(f'Length mask : {mask_paths_for_processing}')
    

    if not image_paths_for_processing:
        print("Error: No images found in the segmentation image directory. Exiting.")
        return
    if Config.RUN_SEGMENTATION and not mask_paths_for_processing:
        print("Warning: Segmentation is enabled but no masks found. Segmentation metrics will not be calculated.")
    
    all_dice_scores = []
    all_true_classes = []
    all_predicted_classes = []

    print(f"Total images to process: {len(image_paths_for_processing)}")

    for img_path in tqdm(image_paths_for_processing, desc="Processing images"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        original_h, original_w = img_np.shape[:2]

        # --- Segmentation Inference ---
        if Config.RUN_SEGMENTATION and seg_model is not None:
            mask_path = os.path.join(Config.SEG_MASK_DIR, base_name + '_OD.tif')
            #print(f'Processing: {mask_path}')
            if os.path.exists(mask_path):
                # Get all patches for segmentation
                patches_with_coords, padded_h, padded_w = get_patches_and_coords(img_np, Config.ORIGINAL_CROP_SIZE)
                
                predicted_patches_for_stitching = []
                for patch_np, (y_start, x_start) in patches_with_coords:
                    patch_tensor = preprocess_transform(Image.fromarray(patch_np)).unsqueeze(0).to(Config.DEVICE)
                    with torch.no_grad():
                        output_binary_512 = (torch.sigmoid(seg_model(patch_tensor)).squeeze().cpu().numpy() > 0.5)
                        predicted_patches_for_stitching.append(((output_binary_512 * 255).astype(np.uint8), (y_start, x_start)))

                # Stitch all patches back
                stitched_pred = stitch_all_patches_back(
                    predicted_patches_for_stitching,
                    original_h, original_w,
                    padded_h, padded_w
                )
                
                gt_mask = (np.array(Image.open(mask_path).convert("L")) > 10).astype(np.uint8)
                #print(f'gt_mask: {gt_mask.shape} & {np.unique(gt_mask)} || stitched_pred: {stitched_pred.shape} & {np.unique(stitched_pred)}')
                dice = dice_score(stitched_pred > 127, gt_mask)
                all_dice_scores.append(dice)

                # Save visualization
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                axs[0].imshow(img_np)
                axs[0].set_title("Original Image")
                axs[1].imshow(gt_mask, cmap='gray')
                axs[1].set_title("Ground Truth Mask")
                axs[2].imshow(stitched_pred, cmap='gray')
                axs[2].set_title(f"Prediction - Dice: {dice:.3f}")
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(Config.OUTPUT_DIR, f"{base_name}_seg.png"))
                plt.close()
            else:
                print(f"Warning: Segmentation mask not found for {base_name}. Skipping segmentation for this image.")


        # --- Classification Inference ---
        if Config.RUN_CLASSIFICATION and cls_model is not None:
            img_path_classif = os.path.join(Config.CLASSIF_IMG_DIR, os.path.basename(img_path))
            if os.path.exists(img_path_classif):
                cls_img = Image.open(img_path_classif).convert("RGB")
                
                # Apply transformations (entire image, not patched, as per your original classification logic)
                input_tensor_cls = preprocess_transform(cls_img).unsqueeze(0).to(Config.DEVICE)

                with torch.no_grad():
                    logits = cls_model(input_tensor_cls)
                    predicted_class = torch.argmax(logits, dim=1).item()
                
                ground_truth_class = classification_ground_truths.get(base_name, None)

                if ground_truth_class is not None:
                    all_true_classes.append(ground_truth_class)
                    all_predicted_classes.append(predicted_class)
                    
                    with open(os.path.join(Config.OUTPUT_DIR, f"{base_name}_class.txt"), 'w') as f:
                        f.write(f"Predicted Class: {predicted_class}\n")
                        f.write(f"Ground Truth Class: {ground_truth_class}\n")
                        f.write(f"Correct Classification: {predicted_class == ground_truth_class}\n")
                    
                    print(f"\n--- Classification Results for {base_name} ---")
                    print(f"Predicted Class: {predicted_class}")
                    print(f"Ground Truth Class: {ground_truth_class}")
                    print(f"Correct: {predicted_class == ground_truth_class}")
                    print("-" * 40)
                else:
                    with open(os.path.join(Config.OUTPUT_DIR, f"{base_name}_class.txt"), 'w') as f:
                        f.write(f"Predicted Class: {predicted_class}\n")
                        f.write("Ground Truth Class: Not Found\n")
                    print(f"\n--- Classification Results for {base_name} (Ground Truth Not Found) ---")
                    print(f"Predicted Class: {predicted_class}")
                    print("-" * 40)
            else:
                print(f"Warning: Classification image not found for {base_name} in {Config.CLASSIF_IMG_DIR}. Skipping classification for this image.")


# ============ FINAL REPORT ============
    if Config.RUN_SEGMENTATION:
        if all_dice_scores:
            avg_dice = np.mean(all_dice_scores)
            print(f"\n__ Average Dice over Test Set (Segmentation): {avg_dice:.4f}")
        else:
            print("\nNo Dice scores calculated (no segmentation images processed or masks found).")

    if Config.RUN_CLASSIFICATION:
        if all_true_classes and all_predicted_classes:
            if len(all_true_classes) == len(all_predicted_classes):
                accuracy = accuracy_score(all_true_classes, all_predicted_classes)
                print(f"\n__ Classification Accuracy: {accuracy:.4f}")
                print("\n__ Classification Report:")
                target_names = [f'Grade {i}' for i in range(Config.CLS_NUM_CLASSES)]
                print(classification_report(all_true_classes, all_predicted_classes, target_names=target_names, zero_division='warn'))
            else:
                print("\nWarning: Mismatch in length of true and predicted class lists. Cannot calculate full classification metrics.")
        else:
            print("\nNo classification results to report (missing ground truth or predictions).")

if __name__ == "__main__":
    run_inference()

'''
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms, models
import segmentation_models_pytorch as smp

from utils.metrics import dice_score

# ============ CONFIG ============
RUN_SEGMENTATION = True
RUN_CLASSIFICATION = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = '/home/nilgiri/Downloads/archive/A20Segmentation/Segmentation/OriginalImages/TestingSet'
MASK_DIR = '/home/nilgiri/Downloads/archive/A20Segmentation/Segmentation/AllSegmentationGroundtruths/TestingSet/OpticDisc'
OUTPUT_DIR = 'test_results/baseline'
CLS_BASELINE_DIR = 'outputs/baseline_models'
SEG_BASELINE_DIR = 'outputs/seg_baseline_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ MODELS ============
if RUN_SEGMENTATION:
    seg_model_path = sorted(glob.glob(os.path.join(SEG_BASELINE_DIR, 'seg_best_model_*.pth')))[-1]
    seg_model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(DEVICE)
    seg_model.load_state_dict(torch.load(seg_model_path, map_location=DEVICE))
    seg_model.eval()
    print(f"Loaded segmentation model from {seg_model_path}")

if RUN_CLASSIFICATION:
    cls_model_path = sorted(glob.glob(os.path.join(CLS_BASELINE_DIR, 'cls_best_model_*.pth')))[-1]
    cls_model = models.resnet18(pretrained=False)
    cls_model.fc = torch.nn.Linear(cls_model.fc.in_features, 5)
    cls_model = cls_model.to(DEVICE)
    cls_model.load_state_dict(torch.load(cls_model_path, map_location=DEVICE))
    cls_model.eval()
    print(f"Loaded classification model from {cls_model_path}")

# ============ TRANSFORMS ============
resize = transforms.Resize((512, 512))
to_tensor = transforms.ToTensor()

# ============ FUNCTIONS ============
def get_four_corners(img_np, crop_size=2048):
    h, w = img_np.shape[:2]
    crops = [
        img_np[0:crop_size, 0:crop_size],
        img_np[0:crop_size, w - crop_size:w],
        img_np[h - crop_size:h, 0:crop_size],
        img_np[h - crop_size:h, w - crop_size:w]
    ]
    return crops

def stitch_back(crops, h, w, crop_size=2048):
    stitched = np.zeros((h, w), dtype=np.uint8)
    stitched[0:crop_size, 0:crop_size] = crops[0]
    stitched[0:crop_size, w - crop_size:w] = crops[1]
    stitched[h - crop_size:h, 0:crop_size] = crops[2]
    stitched[h - crop_size:h, w - crop_size:w] = crops[3]
    return stitched

# ============ INFERENCE ============
image_paths = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))
mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, '*.tif')))
dice_scores = []

for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # --- Segmentation ---
    if RUN_SEGMENTATION:
        pred_crops = []
        gt_mask = np.array(Image.open(mask_path).convert("L")) > 127

        for crop_np in get_four_corners(img_np):
            crop_tensor = to_tensor(resize(Image.fromarray(crop_np))).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = torch.sigmoid(seg_model(crop_tensor)).squeeze().cpu().numpy() > 0.5
                pred_crops.append((output * 255).astype(np.uint8))

        stitched_pred = stitch_back(pred_crops, h, w)
        dice = dice_score(stitched_pred > 127, gt_mask)
        dice_scores.append(dice)

        # Save visualization
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
        plt.savefig(os.path.join(OUTPUT_DIR, f"{base_name}_seg.png"))
        plt.close()

    # --- Classification ---
    if RUN_CLASSIFICATION:
        resized = resize(Image.fromarray(img_np))
        tensor = to_tensor(resized).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = cls_model(tensor)
            pred_class = torch.argmax(logits, dim=1).item()
        with open(os.path.join(OUTPUT_DIR, f"{base_name}_class.txt"), 'w') as f:
            f.write(f"Predicted Class: {pred_class}\n")

# ============ FINAL REPORT ============
if RUN_SEGMENTATION:
    avg_dice = np.mean(dice_scores)
    print(f"\n__ Average Dice over Test Set: {avg_dice:.4f}")
'''
