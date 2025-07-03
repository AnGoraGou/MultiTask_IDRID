import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class ClassificationDataset(Dataset):
    def __init__(self, image_dir, label_csv, joint_transform=None):
        self.image_dir = image_dir
        self.joint_transform = joint_transform

        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.labels_df = pd.read_csv(label_csv)
        self.labels_df.columns = [col.strip() for col in self.labels_df.columns]
        self.labels_df['Image name'] = self.labels_df['Image name'].str.strip()
        self.grade_map = dict(zip(self.labels_df['Image name'], self.labels_df['Retinopathy grade']))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        base_name = os.path.splitext(image_name)[0]

        img_path = os.path.join(self.image_dir, image_name)
        image = Image.open(img_path).convert('RGB')

        if self.joint_transform:
            image, _ = self.joint_transform(image, [])

        label = self.grade_map.get(base_name, -1)
        return image, label
