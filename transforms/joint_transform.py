from torchvision import transforms
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For full reproducibility (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



class JointTransform:
    def __init__(self, crop_size=1024, resize=(512, 512), hflip=True, vflip=True, color_jitter=True, rotation=True):
        self.crop_size = crop_size
        self.resize = resize
        self.hflip = hflip
        self.vflip = vflip
        self.color_jitter = color_jitter
        self.rotation = rotation

        # Define color jitter with mild settings
        self.color_jitter_transform = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
        )

    def __call__(self, image, masks):
        # Random crop
        '''
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        masks = [TF.crop(mask, i, j, h, w) for mask in masks]
        '''
        # Optional horizontal flip
        if self.hflip and random.random() > 0.5:
            image = TF.hflip(image)
            masks = [TF.hflip(mask) for mask in masks]

        # Optional vertical flip
        if self.vflip and random.random() > 0.5:
            image = TF.vflip(image)
            masks = [TF.vflip(mask) for mask in masks]

        #for i, m in enumerate(masks):
        #    print(f"Mask {i} unique values after binarize:", np.unique(m))

        # Optional rotation (0_, 90_, 180_, 270_)
        if self.rotation:
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                image = TF.rotate(image, angle)
                masks = [TF.rotate(mask, angle) for mask in masks]

        # Optional color jitter (only on image, not on masks)
        if self.color_jitter:
            image = self.color_jitter_transform(image)

        # Resize
        image = TF.resize(image, self.resize)
        masks = [TF.resize(mask, self.resize) for mask in masks]

        # To tensor
        image = TF.to_tensor(image)
        #masks = [TF.to_tensor(mask).squeeze(0) for mask in masks]
        masks = [(TF.to_tensor(mask).squeeze(0) > 0.5).float() for mask in masks]
        #for i, m in enumerate(masks):
        #    print(f"Mask {i} unique values after binarize:", torch.unique(m))

        return image, masks

class JointTransform_preAug:
    def __init__(self, crop_size=1024, resize=(512, 512), hflip=True, vflip=True, color_jitter=True, rotation=True):
        self.crop_size = crop_size
        self.resize = resize
        self.hflip = hflip
        self.vflip = vflip
        self.color_jitter = color_jitter
        self.rotation = rotation

        # Define color jitter with mild settings
        self.color_jitter_transform = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
        )

    def __call__(self, image, mask):
        # Random crop
        '''
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        masks = [TF.crop(mask, i, j, h, w) for mask in masks]
        '''
        # Optional horizontal flip
        if self.hflip and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Optional vertical flip
        if self.vflip and random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        #for i, m in enumerate(masks):
        #    print(f"Mask {i} unique values after binarize:", np.unique(m))

        # Optional rotation (0_, 90_, 180_, 270_)
        if self.rotation:
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # Optional color jitter (only on image, not on masks)
        if self.color_jitter:
            image = self.color_jitter_transform(image)

        # Resize
        image = TF.resize(image, self.resize)
        mask = TF.resize(mask, self.resize)

        # To tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask).squeeze(0)
        #masks = [(TF.to_tensor(mask).squeeze(0) > 0.5).float() for mask in masks]
        #for i, m in enumerate(masks):
        #    print(f"Mask {i} unique values after binarize:", torch.unique(m))

        return image, mask
