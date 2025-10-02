"""
预测脚本 - 适配分类方法
使用训练好的模型对测试图像进行预测
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from monai.networks.nets import UNet
import argparse

# 从 config.py 导入配置
from config import (IMG_SIZE, NUM_CLASSES)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = np.load(img_path)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, img_name


def predict(input_dir, output_dir, model_path):
    """主预测函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = UNet(
        in_channels=1,
        out_channels=NUM_CLASSES,
        spatial_dims=2,
        channels=(32, 64, 128, 256, 320, 320),
        strides=(2, 2, 2, 2, 2),
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("模型加载成功!")
    except FileNotFoundError:
        print(f"错误: 未找到模型文件 {model_path}")
        print("请先运行 train.py 训练模型")
        return
    
    # 准备数据
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        ToTensorV2(),
    ])
    
    # 使用TEST_B_DIR作为输入(粒子轨迹)
    test_dataset = SimpleDataset(input_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print(f"开始预测 {len(test_dataset)} 张图像...")
    
    # 预测
    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="预测进度"):
            images = images.to(device, dtype=torch.float32)
            outputs = model(images)  # (B, NUM_CLASSES, H, W)
            
            # 获取每个像素的预测类别
            pred_classes = outputs.argmax(dim=1)  # (B, H, W)
            
            # 转换为连续值 [0, 1]
            pred_values = pred_classes.float() / NUM_CLASSES
            
            # 保存每张图像
            for i, img_name in enumerate(img_names):
                pred_np = pred_values[i].cpu().numpy()
                
                # 保存为.npy文件
                save_path = os.path.join(output_dir, f"pred_{img_name}")
                np.save(save_path, pred_np)
                
                # 可选: 保存为图像以便可视化
                save_img_path = os.path.join(output_dir, f"pred_{img_name.replace('.npy', '.png')}")
                plt.imsave(save_img_path, pred_np, cmap='viridis')
    
    print(f"预测完成! 结果保存在: {output_dir}")


def visualize_comparison(input_dir, output_dir, target_dir, num_samples=5):
    """
    可视化对比: 输入轨迹图 vs 真实势能图 vs 预测势能图
    """
    import glob
    
    print("\n生成对比可视化...")
    
    # 获取文件列表
    input_files = sorted(glob.glob(os.path.join(input_dir, '*.npy')))[:num_samples]
    
    if not input_files:
        print(f"未找到测试文件在: {input_dir}")
        return
    
    comparison_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    
    for input_path in input_files:
        filename = os.path.basename(input_path)
        
        # 加载图像
        input_img = np.load(input_path)
        
        # 尝试加载真实标签
        if target_dir:
            target_path = os.path.join(target_dir, filename)
            if os.path.exists(target_path):
                target_img = np.load(target_path)
            else:
                target_img = None
        else:
            target_img = None

        # 加载预测结果
        pred_path = os.path.join(output_dir, f"pred_{filename}")
        if os.path.exists(pred_path):
            pred_img = np.load(pred_path)
        else:
            print(f"未找到预测文件: {pred_path}")
            continue
        
        # 创建对比图
        if target_img is not None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle(f'Comparison for {filename}', fontsize=16)

            # 1. 输入轨迹
            axes[0, 0].imshow(input_img, cmap='gray')
            axes[0, 0].set_title('Input: Particle Trajectory')
            axes[0, 0].axis('off')
            
            # 2. 真实势能
            im1 = axes[0, 1].imshow(target_img, cmap='viridis')
            axes[0, 1].set_title('Ground Truth: Potential')
            axes[0, 1].axis('off')
            fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

            # 3. 预测势能
            im2 = axes[1, 0].imshow(pred_img, cmap='viridis', vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])
            axes[1, 0].set_title('Prediction: Potential')
            axes[1, 0].axis('off')
            fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

            # 4. 计算并绘制相对误差
            epsilon = 1e-8
            relative_error = np.abs(pred_img - target_img) / (np.abs(target_img) + epsilon) * 100
            
            im3 = axes[1, 1].imshow(relative_error, cmap='inferno', vmin=0, vmax=100)
            axes[1, 1].set_title('Relative Error (%) [Capped at 100%]')
            axes[1, 1].axis('off')
            fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(input_img, cmap='gray')
            axes[0].set_title('Input: Particle Trajectory')
            axes[0].axis('off')
            
            axes[1].imshow(pred_img, cmap='viridis')
            axes[1].set_title('Prediction: Potential')
            axes[1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(comparison_dir, filename.replace('.npy', '_comparison.png'))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"保存对比图: {save_path}")
    
    print(f"对比可视化完成! 保存在: {comparison_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用训练好的U-Net模型进行预测和评估。")
    parser.add_argument('--input-dir', type=str, required=True, help='包含输入.npy文件的目录 (例如, 粒子轨迹数据)。')
    parser.add_argument('--output-dir', type=str, required=True, help='保存预测结果的目录。')
    parser.add_argument('--model-path', type=str, required=True, help='训练好的模型文件 (.pth) 的路径。')
    parser.add_argument('--target-dir', type=str, default=None, help='(可选) 包含真实标签.npy文件的目录，用于生成对比图。')
    parser.add_argument('--num-samples', type=int, default=10, help='用于可视化的样本数量。')

    args = parser.parse_args()

    # 执行预测
    predict(args.input_dir, args.output_dir, args.model_path)
    
    # 如果提供了目标目录，则生成对比可视化
    if args.target_dir:
        try:
            visualize_comparison(args.input_dir, args.output_dir, args.target_dir, args.num_samples)
        except Exception as e:
            print(f"可视化时出错: {e}")
