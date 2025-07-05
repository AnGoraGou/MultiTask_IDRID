#!/usr/bin/env python
# ------------------------------------------------------------
#  baseline_inference_clean.py
#  • Patch-wise batched segmentation & classification
#  • GPU-friendly
#  • Robust CSV / path handling
#  • Summary metrics + per-image logs
# ------------------------------------------------------------
from __future__ import annotations

import json, warnings, glob
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms, models
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights

# ------------------------------------------------------------------
# 1. CONFIG  – all hard-coded paths live here, rooted in the script dir
# ------------------------------------------------------------------
class Config:
    # toggles
    RUN_SEGMENTATION   = False
    RUN_CLASSIFICATION = True

    # device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # project root (folder that holds this file)
    ROOT = Path(__file__).resolve().parent

    # ------------- segmentation -------------
    SEG_IMG_DIR   = ROOT.parent / 'A20Segmentation' / 'Segmentation' / 'OriginalImages' / 'TestingSet'
    SEG_MASK_DIR  = ROOT.parent / 'A20Segmentation' / 'Segmentation' / 'AllSegmentationGroundtruths' / 'TestingSet' / 'OpticDisc'
    SEG_MODEL_DIR = ROOT / 'outputs' / 'seg_baseline_models'

    # ------------- classification -----------
    CLS_IMG_DIR   = ROOT.parent / 'B20Disease20Grading' / 'DiseaseGrading' / 'OriginalImages' / 'TestingSet'
    CLS_GT_CSV    = ROOT.parent / 'B20Disease20Grading' / 'DiseaseGrading' / 'Groundtruths' / 'IDRiD_Disease_Grading_Testing_Labels.csv'
    CLS_MODEL_DIR = ROOT / 'outputs' / 'baseline_models'

    # ------------- outputs ------------------
    OUTPUT_DIR    = ROOT / 'outputs' / 'test_results_refactored'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # image sizes
    ORIGINAL_CROP_SIZE = 2048
    MODEL_INPUT_SIZE   = 512

    # model hyper-params
    SEG_ENCODER_NAME    = 'resnet18'
    SEG_ENCODER_WEIGHTS = 'imagenet'
    SEG_OUT_CHANNELS    = 1
    CLS_NUM_CLASSES     = 5

    # classification aggregation: 'vote' or 'softmax'
    AGGREGATE = 'vote'


# ------------------------------------------------------------------
# 2. SIMPLE UTILS
# ------------------------------------------------------------------
to_model_tensor = transforms.Compose([
    transforms.Resize((Config.MODEL_INPUT_SIZE, Config.MODEL_INPUT_SIZE)),
    transforms.ToTensor()
])

def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """Binary Dice."""
    pred, gt = pred.astype(float), gt.astype(float)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    return (2 * inter + eps) / (union + eps)

# ---------- patch helpers ----------
def extract_patches(im: np.ndarray,
                    patch: int = Config.ORIGINAL_CROP_SIZE
                   ) -> Tuple[List[np.ndarray], List[Tuple[int,int]], int, int]:
    """Return non-overlapping patches, their coords, and padded sizes."""
    h, w = im.shape[:2]
    pad_h = (patch - h % patch) % patch
    pad_w = (patch - w % patch) % patch
    padded = np.pad(im, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    ph, pw = padded.shape[:2]

    patches, coords = [], []
    for y in range(0, ph, patch):
        for x in range(0, pw, patch):
            patches.append(padded[y:y+patch, x:x+patch])
            coords.append((y, x))
    return patches, coords, ph, pw

def stitch_patches(pred_patches: List[np.ndarray], coords: List[Tuple[int,int]],
                   out_h: int, out_w: int, padded_h: int, padded_w: int,
                   patch: int = Config.ORIGINAL_CROP_SIZE) -> np.ndarray:
    """Stitch 512-sized predicted patches (uint8) back to original canvas."""
    canvas = np.zeros((padded_h, padded_w), dtype=np.uint8)
    for p, (y, x) in zip(pred_patches, coords):
        big = np.array(Image.fromarray(p).resize((patch, patch), Image.NEAREST))
        canvas[y:y+patch, x:x+patch] = big
    return canvas[:out_h, :out_w]

# ------------------------------------------------------------------
# 3. MODEL LOADING
# ------------------------------------------------------------------
def load_segmentation_model() -> nn.Module | None:
    if not Config.SEG_MODEL_DIR.exists():
        warnings.warn('[SEG] model dir not found'); return None
    ckpts = sorted(Config.SEG_MODEL_DIR.glob('seg_best_model_*.pth'))
    if not ckpts:
        warnings.warn('[SEG] no checkpoint'); return None
    ckpt = ckpts[-1]
    model = smp.Unet(
        encoder_name=Config.SEG_ENCODER_NAME,
        encoder_weights=Config.SEG_ENCODER_WEIGHTS,
        in_channels=3,
        classes=Config.SEG_OUT_CHANNELS
    ).to(Config.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=Config.DEVICE))
    model.eval()
    print(f'[SEG] loaded {ckpt.name}')
    return model

def load_classification_model() -> nn.Module | None:
    if not Config.CLS_MODEL_DIR.exists():
        warnings.warn('[CLS] model dir not found'); return None
    ckpts = sorted(Config.CLS_MODEL_DIR.glob('cls_best_model_*.pth'))
    if not ckpts:
        warnings.warn('[CLS] no checkpoint'); return None
    ckpt = ckpts[-1]
    #model = models.resnet18(pretrained=False)
    weights = ResNet18_Weights.IMAGENET1K_V1  # or None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, Config.CLS_NUM_CLASSES)
    model.to(Config.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=Config.DEVICE))
    model.eval()
    print(f'[CLS] loaded {ckpt.name}')
    return model

# ------------------------------------------------------------------
# 4. GROUND-TRUTH CSV
# ------------------------------------------------------------------
def load_ground_truths() -> Dict[str, int]:
    #print(Config.CLS_GT_CSV)
    try:
        df = pd.read_csv(Config.CLS_GT_CSV)
    except FileNotFoundError:
        warnings.warn('[CLS] ground-truth CSV not found; disabling classification')
        Config.RUN_CLASSIFICATION = False
        return {}
    need = {'Image name', 'Retinopathy grade'}
    if not need.issubset(df.columns):
        raise ValueError(f'CSV missing cols: {need - set(df.columns)}')
    gt = {row['Image name'].split('.')[0]: int(row['Retinopathy grade'])
          for _, row in df.iterrows()}
    print(f'[CLS] loaded {len(gt)} GT labels')
    return gt

# ------------------------------------------------------------------
# 5. INFERENCE ROUTINES
# ------------------------------------------------------------------
def infer_segmentation(model: nn.Module,
                       img_np: np.ndarray,
                       base: str) -> float | None:
    """Return Dice score (or None)."""
    if model is None: return None
    patches, coords, ph, pw = extract_patches(img_np)
    batch = torch.stack([to_model_tensor(Image.fromarray(p)) for p in patches]
                       ).to(Config.DEVICE)
    with torch.no_grad():
        preds = (torch.sigmoid(model(batch)) > 0.5).float()
    pred_bytes = [(pred.squeeze() * 255).byte().cpu().numpy() for pred in preds]
    stitched = stitch_patches(pred_bytes, coords,
                              *img_np.shape[:2], ph, pw)

    mask_path = Config.SEG_MASK_DIR / f'{base}_OD.tif'
    #print(mask_path)
    #exit()
    if not mask_path.exists():
        warnings.warn(f'[SEG] no GT mask for {base}')
        return None
    gt = (np.array(Image.open(mask_path).convert('L')) > 10).astype(np.uint8)
    dsc = dice_score(stitched > 127, gt)

    # quick visual
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    ax[0].imshow(img_np); ax[0].set_title('Image')
    ax[1].imshow(gt, cmap='gray'); ax[1].set_title('GT')
    ax[2].imshow(stitched, cmap='gray'); ax[2].set_title(f'Pred (Dice={dsc:.3f})')
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / f'{base}_seg.png')
    plt.close()
    return dsc

def aggregate_patch_predictions(logits: torch.Tensor) -> int:
    if Config.AGGREGATE == 'softmax':
        probs = torch.softmax(logits, 1).mean(0)
        return int(probs.argmax().item())
    # vote
    labels = torch.argmax(logits, 1).cpu().numpy()
    return int(np.bincount(labels).argmax())

def infer_classification(model: nn.Module,
                         img_np: np.ndarray,
                         base: str,
                         gt_map: Dict[str, int],
                         true_cls: List[int],
                         pred_cls: List[int]):
    if model is None: return
    patches, _, _, _ = extract_patches(img_np)
    if not patches:
        warnings.warn(f'[CLS] no patches for {base}'); return
    batch = torch.stack([to_model_tensor(Image.fromarray(p)) for p in patches]
                       ).to(Config.DEVICE)
    with torch.no_grad():
        logits = model(batch)
    pred = aggregate_patch_predictions(logits)

    gt = gt_map.get(base)
    # write log
    (Config.OUTPUT_DIR / f'{base}_class.txt').write_text(
        json.dumps({'pred': pred, 'gt': gt}, indent=2)
    )
    if gt is not None:
        true_cls.append(gt)
        pred_cls.append(pred)

# ------------------------------------------------------------------
# 6. MAIN
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 6. MAIN – separate loops for SEG and CLS
# ------------------------------------------------------------------

def main():
    seg_model = load_segmentation_model() if Config.RUN_SEGMENTATION   else None
    cls_model = load_classification_model() if Config.RUN_CLASSIFICATION else None
    gt_map    = load_ground_truths()        if Config.RUN_CLASSIFICATION else {}

    # ---------- 6-A  Segmentation loop ----------
    dice_scores = []
    if seg_model:
        seg_imgs = sorted(Config.SEG_IMG_DIR.glob('*.jpg'))
        if not seg_imgs:
            print('[SEG] No test images found.')
        for p in tqdm(seg_imgs, desc='Segmentation'):
            base = p.stem          # keep the .lower() because masks use same base
            try:
                img_np = np.array(Image.open(p).convert('RGB'))
            except Exception as e:
                warnings.warn(f'[SEG] Cannot read {p.name}: {e}')
                continue
            d = infer_segmentation(seg_model, img_np, base)
            if d is not None:
                dice_scores.append(d)

    # ---------- 6-B  Classification loop ----------
    true_cls, pred_cls = [], []
    if cls_model:
        cls_imgs = sorted(Config.CLS_IMG_DIR.glob('*.jpg'))
        if not cls_imgs:
            print('[CLS] No test images found.')
        for p in tqdm(cls_imgs, desc='Classification'):
            base = p.stem                  # NOTE: *no* .lower() – the CSV keys keep case
            try:
                img_np = np.array(Image.open(p).convert('RGB'))
            except Exception as e:
                warnings.warn(f'[CLS] Cannot read {p.name}: {e}')
                continue
            infer_classification(cls_model, img_np, base, gt_map,
                                 true_cls, pred_cls)

    # ---------- 6-C  Summary ----------
    if dice_scores:
        print(f'\n[SEG] Mean Dice: {np.mean(dice_scores):.4f}'
              f' on {len(dice_scores)} images')

    if true_cls:
        acc = accuracy_score(true_cls, pred_cls)
        print(f'\n[CLS] Accuracy : {acc:.4f}')
        print(classification_report(
            true_cls, pred_cls,
            target_names=[f'Grade {i}' for i in range(Config.CLS_NUM_CLASSES)],
            zero_division='warn'
        ))
    elif cls_model:
        print('\n[CLS] No test images had ground-truth labels ⇒ metrics skipped.')


'''
def main():
    #print(Config.SEG_IMG_DIR)
    
    seg_model = load_segmentation_model() if Config.RUN_SEGMENTATION else None
    cls_model = load_classification_model() if Config.RUN_CLASSIFICATION else None
    gt_map    = load_ground_truths()        if Config.RUN_CLASSIFICATION else {}

    img_paths = sorted(Config.SEG_IMG_DIR.glob('*.jpg'))
    if not img_paths:
        print('[MAIN] no test images found – nothing to do'); return

    dice_scores, true_cls, pred_cls = [], [], []
    for p in tqdm(img_paths, desc='Images'):
        base = p.stem
        try:
            img_np = np.array(Image.open(p).convert('RGB'))
        except Exception as e:
            warnings.warn(f'Cannot read {p.name}: {e}')
            continue

        # segmentation
        if seg_model:
            d = infer_segmentation(seg_model, img_np, base)
            if d is not None: dice_scores.append(d)

        # classification
        if cls_model:
            infer_classification(cls_model, img_np, base,
                                 gt_map, true_cls, pred_cls)

    # ---------- summary ----------
    if dice_scores:
        print(f'\n[SEG] Mean Dice  : {np.mean(dice_scores):.4f} '
              f'on {len(dice_scores)} images')

    if true_cls:
        acc = accuracy_score(true_cls, pred_cls)
        print(f'\n[CLS] Accuracy   : {acc:.4f}')
        print(classification_report(
            true_cls, pred_cls,
            target_names=[f'Grade {i}' for i in range(Config.CLS_NUM_CLASSES)],
            zero_division='warn'
        ))
    elif cls_model:
        print('\n[CLS] No images had ground-truth labels → metrics skipped.')
'''
if __name__ == '__main__':
    main()
