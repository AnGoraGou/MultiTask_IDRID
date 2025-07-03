import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dirs, joint_transform=None):
        self.image_dir = image_dir
        self.mask_dirs = mask_dirs
        self.joint_transform = joint_transform

        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.suffix_map = {
            # 'Microaneurysms': '_MA.tif',
            # 'Haemorrhages': '_HE.tif',
            # 'HardExudates': '_EX.tif',
            # 'SoftExudates': '_SE.tif',
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
            suffix = self.suffix_map[lesion]
            mask_file = base_name + suffix
            mask_path = os.path.join(mask_dir, mask_file)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
            else:
                mask = Image.new('L', image.size)

            masks.append(mask)

        if self.joint_transform:
            image, masks = self.joint_transform(image, masks)

        masks_tensor = torch.stack(masks, dim=0)
        return image, masks_tensor
