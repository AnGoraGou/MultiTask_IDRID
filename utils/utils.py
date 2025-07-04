class JointTransform:
    def __init__(self, mode='train', crop_size=1024, resize=(512, 512), hflip=True, vflip=True, color_jitter=True, rotation=True):
        assert mode in ['train', 'val'], "Mode must be 'train' or 'val'"
        self.mode = mode
        self.crop_size = crop_size
        self.resize = resize
        self.hflip = hflip
        self.vflip = vflip
        self.color_jitter = color_jitter
        self.rotation = rotation

        self.color_jitter_transform = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
        )

    def __call__(self, image, masks):
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        masks = [TF.crop(mask, i, j, h, w) for mask in masks]
        
        if self.mode == 'train':
            # Random crop
            #i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            #image = TF.crop(image, i, j, h, w)
            #masks = [TF.crop(mask, i, j, h, w) for mask in masks]

            if self.hflip and random.random() > 0.5:
                image = TF.hflip(image)
                masks = [TF.hflip(mask) for mask in masks]

            if self.vflip and random.random() > 0.5:
                image = TF.vflip(image)
                masks = [TF.vflip(mask) for mask in masks]

            if self.rotation:
                angle = random.choice([0, 90, 180, 270])
                image = TF.rotate(image, angle)
                masks = [TF.rotate(mask, angle) for mask in masks]

            if self.color_jitter:
                image = self.color_jitter_transform(image)

        # Resize for both train and val
        image = TF.resize(image, self.resize)
        masks = [TF.resize(mask, self.resize) for mask in masks]

        # To Tensor and binarize mask
        image = TF.to_tensor(image)
        masks = [(TF.to_tensor(mask).squeeze(0) > 0.5).float() for mask in masks]

        return image, masks

        
        
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dirs, joint_transform=None):
        self.image_dir = image_dir
        self.mask_dirs = mask_dirs
        self.joint_transform = joint_transform

        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.suffix_map = {
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
            suffix = self.suffix_map.get(lesion, '.png')
            mask_path = os.path.join(mask_dir, base_name + suffix)

            mask = Image.open(mask_path).convert('L') if os.path.exists(mask_path) else Image.new('L', image.size)
            masks.append(mask)

        if self.joint_transform:
            image, masks = self.joint_transform(image, masks)

        masks_tensor = torch.stack(masks, dim=0)  # [1, H, W]
        return image, masks_tensor.squeeze(0)     # [3, H, W], [H, W]




class DiceLoss_fromLogits(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss_fromLogits, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()

class DiceBCE_fromLogits(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None, smooth=1e-6):
        super(DiceBCE_fromLogits, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss_fromLogits(smooth=smooth)

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
