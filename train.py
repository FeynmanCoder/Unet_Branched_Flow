import os
import sys

from monai.networks.nets import UNet

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F

# 从 config.py 导入配置
from config import (IMG_SIZE, NUM_CLASSES, TRAIN_A_DIR as TRAIN_IMG_DIR, 
                    TRAIN_B_DIR as TRAIN_MASK_DIR, VALIDATION_A_DIR as VAL_IMG_DIR, 
                    VALIDATION_B_DIR as VAL_MASK_DIR, BATCH_SIZE, NUM_WORKERS, 
                    LEARNING_RATE, EPOCHS, BEST_MODEL_PATH)


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = np.load(img_path)
        mask = np.load(mask_path)
        
        # 将连续的势能值离散化为NUM_CLASSES个类别
        # 这是关键改进:将回归问题转换为分类问题,避免模型直接复制输入特征
        mask = (mask * NUM_CLASSES)
        mask[mask >= NUM_CLASSES] = NUM_CLASSES - 1
        mask[mask < 0] = 0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()


def dice_loss_multiclass(pred, target, smooth=1e-5):
    """
    多分类Dice Loss - 帮助模型更好地学习分割边界
    """
    pred = F.softmax(pred, dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    
    pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
    one_hot = one_hot.contiguous().view(one_hot.size(0), one_hot.size(1), -1)
    
    intersection = (pred * one_hot).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2) + one_hot.sum(dim=2) + smooth)
    return 1 - dice.mean()


ce_loss = nn.CrossEntropyLoss()


def combined_loss(pred, target):
    """组合损失函数:交叉熵 + Dice Loss"""
    ce = ce_loss(pred, target)
    dice = dice_loss_multiclass(pred, target)
    return ce + dice


if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])

    print('准备数据集')
    train_dataset = SegmentationDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=transform
    )

    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        ToTensorV2(),
    ])
    
    val_dataset = SegmentationDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print('开始训练')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(
        in_channels=1,
        out_channels=NUM_CLASSES,
        spatial_dims=2,
        channels=(32, 64, 128, 256, 320, 320),
        strides=(2, 2, 2, 2, 2),
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    num_epochs = EPOCHS
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        for images, masks in train_bar:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.long)
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f'Model saved at epoch {epoch + 1} with val loss: {val_loss:.4f}')

    print('finish')
