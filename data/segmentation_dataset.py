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

#This will be used to load data from pre-augmented images and masks paired stored 
class SegmentationDataset_preAug(Dataset):
    def __init__(self, image_dir, mask_dirs, joint_transform=None):
        self.image_dir = image_dir
        self.mask_dirs = mask_dirs
        self.joint_transform = joint_transform
        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        base_name = os.path.splitext(image_name)[0]
        img_path = os.path.join(self.image_dir, image_name)
        image = Image.open(img_path).convert('RGB')

        mask_path = img_path.replace('images','masks').replace('jpg','png')


        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            Print(f'Error 505!!! {mask_path} does not exists.')



        #print(f'Unique: {np.unique(mask)} of {mask_path} /n {img_path}')

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        #masks_tensor = torch.stack(masks, dim=0)
        return image, mask
