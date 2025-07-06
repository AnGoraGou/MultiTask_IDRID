
#inference.py  _  Multi-task IDRiD test-time runner
#
#  * Reuses the training-time datasets / transforms / model
#  * One or two test sets: segmentation &/or classification
#  * Patch-wise inference with router-weighted fusion
#  * Optional Dice/accuracy if ground-truth is supplied
#  * Results _ CSV  (+ PNG overlays for _interesting_ cases)
# ------------------------------------------------------------
import argparse, json, warnings
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# __ project imports _________________________________________
from models.model import MultiTaskNet                      # training model
from data.segmentation_dataset import SegmentationDataset  # _ reuse pre-proc
from data.classification_dataset import ClassificationDataset
from transforms.joint_transform import JointTransform
#from utils.metrics import dice_score                       # same metric impl
# if you have an accuracy helper in utils.metrics, import; else fallback:
#try:
#    from utils.metrics import accuracy
#except ImportError:
#    def accuracy(y_true, y_pred):
#        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
#        return (y_true == y_pred).mean()

# _____________________ utilities ___________________________
def extract_patches(img: np.ndarray,
                    patch: int) -> Tuple[List[np.ndarray], List[Tuple[int,int]],
                                         int, int]:
    """Pad to multiple of `patch` and chop into non-overlapping tiles."""
    h, w = img.shape[:2]
    pad_h = (patch - h % patch) % patch
    pad_w = (patch - w % patch) % patch
    padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    ph, pw = padded.shape[:2]
    tiles, coords = [], []
    for y in range(0, ph, patch):
        for x in range(0, pw, patch):
            tiles.append(padded[y:y+patch, x:x+patch])
            coords.append((y, x))
    return tiles, coords, h, w

def stitch_patches(mask_tiles: List[np.ndarray],
                   coords: List[Tuple[int,int]],
                   oh: int, ow: int,
                   patch: int) -> np.ndarray:
    """Back-assemble full-res mask from per-patch predictions."""
    ph = ((oh + patch - 1) // patch) * patch
    pw = ((ow + patch - 1) // patch) * patch
    canvas = np.zeros((ph, pw), dtype=np.uint8)
    for m, (y, x) in zip(mask_tiles, coords):
        m_big = np.array(Image.fromarray(m).resize((patch, patch),
                                                   Image.NEAREST))
        canvas[y:y+patch, x:x+patch] = m_big
    return canvas[:oh, :ow]

def colour_overlay(img: np.ndarray, mask: np.ndarray,
                   alpha: float = 0.4) -> np.ndarray:
    """RGB overlay with the binary mask painted red."""
    overlay = img.copy().astype(np.float32)
    overlay[mask > 0, 0] = 255  # red channel
    return (alpha * overlay + (1-alpha) * img).astype(np.uint8)


import numpy as np

def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Args:
        pred (np.ndarray): Predicted binary mask.
        target (np.ndarray): Ground truth binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice score between 0.0 and 1.0.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

import torch

def accuracy(targets, outputs):
    """
    Compute accuracy from predicted logits and true class labels.

    Args:
        targets (List[int] or Tensor): Ground truth class labels.
        outputs (List[List[float]] or Tensor): Predicted logits or probabilities.

    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    outputs = torch.tensor(outputs)  # Convert list to tensor if needed
    targets = torch.tensor(targets)

    if outputs.dim() == 1:
        # If outputs is 1D, probably already class indices or single logit per sample
        preds = outputs.argmax(dim=0)
    else:
        preds = outputs.argmax(dim=1)

    correct = (preds == targets).sum().item()
    total = targets.size(0)

    return correct / total if total > 0 else 0.0



# _____________________ inference core ______________________
@torch.no_grad()
def infer_one_image(model: nn.Module,
                    img_np: np.ndarray,
                    patch_size: int,
                    input_size: int,
                    device: torch.device,
                    use_amp: bool = True):
    """Returns (stitched_mask, cls_logits_mean, router_weights, patch_logits)."""
    # --- patch _ tensor ----------------------------------------------------
    tiles, coords, oh, ow = extract_patches(img_np, patch_size)
    to_tensor = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
    batch = torch.stack([to_tensor(Image.fromarray(t)) for t in tiles]
                        ).to(device)

    with torch.cuda.amp.autocast(enabled=use_amp):
        out = model(batch, task="both", return_routing=True)
        seg = torch.sigmoid(out['seg_out'])            # [B,1,512,512]
        cls = out['cls_out']                           # [B,5]
        rout = out['routing_weights'][:, 1]                       # take w_cls prob

    # --- segmentation ------------------------------------------------------
    seg_bin = (seg > 0.5).float().squeeze(1)           # [B,512,512]
    mask_tiles = [(m.cpu().byte().numpy()*255) for m in seg_bin]
    stitched_mask = stitch_patches(mask_tiles, coords, oh, ow, patch_size)

    # --- classification fusion --------------------------------------------
    # router-weighted mean (fallback to plain mean if sum == 0)
    weights = rout.cpu().numpy()
    logits = cls.cpu().numpy()
    wsum = weights.sum()
    fused_logits = (logits * weights[:, None]).sum(0) / (wsum + 1e-7)
    return stitched_mask, fused_logits, weights, logits

# _____________________ argparse & main ____________________
import argparse
from pathlib import Path
import torch


def parse_args():
    """CLI flags that are really consumed by inference.py."""
    p = argparse.ArgumentParser(
        description="Patch-wise multi-task inference for IDRiD")

    # __ data inputs _______________________________________________
    p.add_argument("--seg-img-dir", type=Path, default='/workspace/A20Segmentation/Segmentation/OriginalImages/TestingSet',
                   help="Folder with fundus JPGs for optic-disc segmentation")
    p.add_argument("--seg-mask-dir", type=Path, default='/workspace/A20Segmentation/Segmentation/AllSegmentationGroundtruths/TestingSet/OpticDisc',
                   help="GT optic-disc masks (optional - enables Dice)")
    p.add_argument("--cls-img-dir", type=Path, default='/workspace/B20Disease20Grading/DiseaseGrading/OriginalImages/TestingSet',
                   help="Folder with fundus JPGs for DR grading")
    p.add_argument("--gt-csv", type=Path, default='/workspace/B20Disease20Grading/DiseaseGrading/Groundtruths/IDRiD_Disease_Grading_Testing_Labels.csv',
                   help="CSV with ground-truth DR grades (optional)")

    # __ model / runtime ___________________________________________
    p.add_argument("--checkpoint", type=Path, default='/workspace/idrid_project/outputs/models_mtlr/best_mtlr_model_20250706_012633.pth', #required=True,
                   help="Path to *.pth checkpoint from training")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda | cpu | cuda:0 _")
    p.add_argument("--amp/--no-amp", dest="amp", default=True,
                   help="Enable mixed-precision inference")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile for extra speed (PyTorch _ 2.3)")

    # __ model input geometry _____________________________________
    p.add_argument("--patch-size", type=int, default=2048,
                   help="Size of sliding window patch on the original image")
    p.add_argument("--input-size", type=int, default=512,
                   help="Resolution fed to the network (after Resize)")

    # __ outputs / diagnostics ____________________________________
    p.add_argument("--output-dir", type=Path, default=Path("inference_out"),
                   help="Root folder for CSV + overlays")
    p.add_argument("--save-overlays", action="store_true",
                   help="Save PNG overlays for low-confidence or bad-Dice cases")
    p.add_argument("--dice-thresh", type=float, default=0.80,
                   help="Overlay if Dice < threshold (only when GT masks present)")
    p.add_argument("--prob-thresh", type=float, default=0.55,
                   help="Overlay if max class-probability < threshold")

    return p.parse_args()



def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    if args.save_overlays:
        overlay_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)
    # --- model -------------------------------------------------------------
    #model = MultiTaskNet(num_classes=5).to(device)

    model = MultiTaskNet(
            encoder_name="resnet18",
            encoder_weights=None,
            num_classes=5
            ).to(device)


    #ckpt = torch.load(args.checkpoint, map_location=device)
    #model.load_state_dict(ckpt)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    if args.compile:
        model = torch.compile(model)

    # --- transforms (identical resize / norm as training, but no aug) ------
    joint_tf = JointTransform(crop_size=2048, resize=(512, 512), hflip=False, vflip=False, color_jitter=False, rotation=False)

    # --- datasets ----------------------------------------------------------
    seg_items, cls_items = [], []

    if args.seg_img_dir:
        seg_items = sorted(Path(args.seg_img_dir).glob("*.jpg"))
        if not seg_items:
            warnings.warn("No segmentation test images found!")

    if args.cls_img_dir:
        cls_items = sorted(Path(args.cls_img_dir).glob("*.jpg"))
        if not cls_items:
            warnings.warn("No classification test images found!")

    # GT label map (if provided)
    gt_map = {}
    if args.gt_csv and args.gt_csv.exists():
        df_gt = pd.read_csv(args.gt_csv)
        gt_map = {row["Image name"].split('.')[0]: int(row["Retinopathy grade"])
                  for _, row in df_gt.iterrows()}

    # --- metrics collectors -----------------------------------------------
    dice_scores = []
    y_true, y_pred = [], []
    records = []

    # helper to maybe compute Dice
    def maybe_dice(pred_mask: np.ndarray, base: str) -> Union[float, None]:
        if not args.seg_mask_dir:
            return None
        gt_path = args.seg_mask_dir / f"{base}_OD.tif"
        if not gt_path.exists():
            return None
        gt = (np.array(Image.open(gt_path).convert("L")) > 10)
        return float(dice_score(pred_mask > 127, gt))

    # --- run segmentation set ---------------------------------------------
    for img_path in tqdm(seg_items, desc="SEG set"):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        mask, fused_logits, w, patch_logits = infer_one_image(
            model, img_np, args.patch_size, args.input_size,
            device, args.amp)

        base = img_path.stem
        cls_id = int(fused_logits.argmax())
        dice = maybe_dice(mask, base)
        if dice is not None:
            dice_scores.append(dice)

        # overlay logic
        if args.save_overlays:
            need = (dice is not None and dice < args.dice_thresh) \
                   or fused_logits.max() < args.prob_thresh
            if need:
                ov = colour_overlay(img_np, mask > 127)
                Image.fromarray(ov).save(overlay_dir / f"{base}.png")

        # collect metrics
        gt_grade = gt_map.get(base)
        if gt_grade is not None:
            y_true.append(gt_grade)
            y_pred.append(cls_id)

        records.append({
            "image":   img_path.name,
            "task":    "seg+cls",
            "grade":   cls_id,
            "prob":    float(fused_logits.max()),
            "dice":    dice,
            "router_w_mean": float(np.mean(w))
        })

    # --- run classification-only set --------------------------------------
    for img_path in tqdm(cls_items, desc="CLS set"):
        if args.seg_img_dir and (args.seg_img_dir / img_path.name).exists():
            # already processed above
            continue
        img_np = np.array(Image.open(img_path).convert("RGB"))
        _, fused_logits, w, _ = infer_one_image(
            model, img_np, args.patch_size, args.input_size,
            device, args.amp)

        base = img_path.stem
        cls_id = int(fused_logits.argmax())

        # overlay logic (classification-only)
        if args.save_overlays and fused_logits.max() < args.prob_thresh:
            ov = colour_overlay(img_np,
                                np.zeros(img_np.shape[:2], dtype=bool))
            Image.fromarray(ov).save(overlay_dir / f"{base}.png")

        gt_grade = gt_map.get(base)
        if gt_grade is not None:
            y_true.append(gt_grade)
            y_pred.append(cls_id)

        records.append({
            "image":   img_path.name,
            "task":    "cls",
            "grade":   cls_id,
            "prob":    float(fused_logits.max()),
            "dice":    None,
            "router_w_mean": float(np.mean(w))
        })

    # --- write CSV ---------------------------------------------------------
    df_out = pd.DataFrame(records)
    csv_path = args.output_dir / "predictions.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df_out)} predictions _ {csv_path}")

    # --- summary metrics ---------------------------------------------------
    if dice_scores:
        print(f"[SEG] Mean Dice = {np.mean(dice_scores):.4f} "
              f"on {len(dice_scores)} images")

    if y_true:
        acc = accuracy(y_true, y_pred)
        print(f"[CLS] Accuracy  = {acc:.4f} "
              f"on {len(y_true)} images with GT")

if __name__ == "__main__":
    main()
