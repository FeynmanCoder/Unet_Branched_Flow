# test_model.py - 测试模型的通道数是否正确

import torch
import config
from model import UNet

def test_model():
    """测试U-Net模型的前向传播"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = UNet(
        n_channels=config.MODEL_N_CHANNELS,
        n_classes=config.MODEL_N_CLASSES,
        bilinear=config.MODEL_BILINEAR,
        depth=config.MODEL_DEPTH,
        base_channels=config.MODEL_BASE_CHANNELS
    ).to(device)
    
    print(f"Model created with parameters:")
    print(f"  - n_channels: {config.MODEL_N_CHANNELS}")
    print(f"  - n_classes: {config.MODEL_N_CLASSES}")
    print(f"  - bilinear: {config.MODEL_BILINEAR}")
    print(f"  - depth: {config.MODEL_DEPTH}")
    print(f"  - base_channels: {config.MODEL_BASE_CHANNELS}")
    
    # 创建测试输入
    batch_size = 2
    test_input = torch.randn(batch_size, config.MODEL_N_CHANNELS, config.IMG_SIZE, config.IMG_SIZE).to(device)
    
    print(f"\nTest input shape: {test_input.shape}")
    
    # 测试前向传播
    try:
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        print(f"Success! Output shape: {output.shape}")
        print(f"Expected output shape: ({batch_size}, {config.MODEL_N_CLASSES}, {config.IMG_SIZE}, {config.IMG_SIZE})")
        
        # 检查输出形状是否正确
        expected_shape = (batch_size, config.MODEL_N_CLASSES, config.IMG_SIZE, config.IMG_SIZE)
        if output.shape == expected_shape:
            print("✓ Output shape is correct!")
            return True
        else:
            print(f"✗ Output shape mismatch. Expected {expected_shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 Model test passed! The channel mismatch issue has been resolved.")
    else:
        print("\n❌ Model test failed. Please check the error messages above.")