"""
运行前环境检查脚本
在训练之前运行此脚本,确保所有配置正确
"""
import os
import sys

def check_imports():
    """检查必需的Python包"""
    print("=" * 60)
    print("检查 Python 包...")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'monai': 'MONAI',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'albumentations': 'Albumentations',
        'tqdm': 'TQDM',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name}: 已安装")
        except ImportError:
            print(f"❌ {name}: 未安装")
            missing.append(package)
    
    if missing:
        print("\n⚠️ 缺少以下包,请安装:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✅ 所有必需的包都已安装!\n")
    return True


def check_cuda():
    """检查CUDA可用性"""
    print("=" * 60)
    print("检查 CUDA...")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用")
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            print(f"   当前 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA 版本: {torch.version.cuda}")
        else:
            print("⚠️ CUDA 不可用, 将使用 CPU 训练")
            print("   (CPU训练会很慢,建议使用GPU)")
    except Exception as e:
        print(f"❌ 检查CUDA时出错: {e}")
    
    print()


def check_config():
    """检查配置文件"""
    print("=" * 60)
    print("检查配置文件...")
    print("=" * 60)
    
    try:
        from config import (DATA_DIR, TRAIN_A_DIR, TRAIN_B_DIR, 
                           VALIDATION_A_DIR, VALIDATION_B_DIR,
                           NUM_CLASSES, IMG_SIZE, BATCH_SIZE, EPOCHS,
                           BEST_MODEL_PATH)
        
        print(f"✅ config.py 导入成功")
        print(f"\n关键配置:")
        print(f"  数据目录: {DATA_DIR}")
        print(f"  类别数量: {NUM_CLASSES}")
        print(f"  图像大小: {IMG_SIZE}")
        print(f"  批次大小: {BATCH_SIZE}")
        print(f"  训练轮数: {EPOCHS}")
        print(f"  模型保存: {BEST_MODEL_PATH}")
        
        return True
    except ImportError as e:
        print(f"❌ 无法导入 config.py: {e}")
        return False
    
    print()


def check_data_dirs():
    """检查数据目录"""
    print("=" * 60)
    print("检查数据目录...")
    print("=" * 60)
    
    try:
        from config import (DATA_DIR, TRAIN_A_DIR, TRAIN_B_DIR,
                           VALIDATION_A_DIR, VALIDATION_B_DIR,
                           TEST_A_DIR, TEST_B_DIR)
        
        dirs_to_check = {
            'DATA_DIR': DATA_DIR,
            'TRAIN_A_DIR (势能/标签)': TRAIN_A_DIR,
            'TRAIN_B_DIR (轨迹/输入)': TRAIN_B_DIR,
            'VALIDATION_A_DIR': VALIDATION_A_DIR,
            'VALIDATION_B_DIR': VALIDATION_B_DIR,
            'TEST_A_DIR': TEST_A_DIR,
            'TEST_B_DIR': TEST_B_DIR,
        }
        
        all_exist = True
        for name, path in dirs_to_check.items():
            if os.path.exists(path):
                npy_files = len([f for f in os.listdir(path) if f.endswith('.npy')])
                print(f"✅ {name}: {npy_files} 个 .npy 文件")
            else:
                print(f"❌ {name}: 不存在")
                print(f"   路径: {path}")
                all_exist = False
        
        if not all_exist:
            print("\n⚠️ 部分数据目录不存在,请检查 config.py 中的路径设置")
            return False
        
        print("\n✅ 所有数据目录都存在!\n")
        return True
        
    except Exception as e:
        print(f"❌ 检查数据目录时出错: {e}")
        return False


def check_data_samples():
    """检查数据样本"""
    print("=" * 60)
    print("检查数据样本...")
    print("=" * 60)
    
    try:
        import numpy as np
        from config import TRAIN_A_DIR, TRAIN_B_DIR
        
        # 获取A和B目录的文件列表
        files_a = sorted([f for f in os.listdir(TRAIN_A_DIR) if f.endswith('.npy')])
        files_b = sorted([f for f in os.listdir(TRAIN_B_DIR) if f.endswith('.npy')])
        
        if not files_a or not files_b:
            print("❌ 训练目录中没有 .npy 文件")
            return False
        
        # 检查文件名是否匹配
        if set(files_a) != set(files_b):
            print("⚠️ trainA 和 trainB 的文件名不完全匹配")
            print(f"   trainA: {len(files_a)} 个文件")
            print(f"   trainB: {len(files_b)} 个文件")
        else:
            print(f"✅ 文件名匹配: {len(files_a)} 对训练样本")
        
        # 检查前3个样本
        print("\n检查前3个样本:")
        for i, fname in enumerate(files_a[:3]):
            path_a = os.path.join(TRAIN_A_DIR, fname)
            path_b = os.path.join(TRAIN_B_DIR, fname)
            
            if os.path.exists(path_b):
                data_a = np.load(path_a)
                data_b = np.load(path_b)
                
                print(f"\n  样本 {i+1}: {fname}")
                print(f"    A (势能) - shape: {data_a.shape}, range: [{data_a.min():.4f}, {data_a.max():.4f}]")
                print(f"    B (轨迹) - shape: {data_b.shape}, range: [{data_b.min():.4f}, {data_b.max():.4f}]")
                
                # 检查数据范围
                if data_a.min() < 0 or data_a.max() > 1:
                    print(f"    ⚠️ A 数据范围异常,应该在 [0, 1] 之间")
                if data_b.min() < 0 or data_b.max() > 1:
                    print(f"    ⚠️ B 数据范围异常,应该在 [0, 1] 之间")
        
        print("\n✅ 数据样本检查完成!\n")
        return True
        
    except Exception as e:
        print(f"❌ 检查数据样本时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_train_script():
    """检查训练脚本"""
    print("=" * 60)
    print("检查训练脚本...")
    print("=" * 60)
    
    if not os.path.exists('train.py'):
        print("❌ train.py 不存在")
        return False
    
    print("✅ train.py 存在")
    
    # 尝试导入检查语法
    try:
        import py_compile
        py_compile.compile('train.py', doraise=True)
        print("✅ train.py 语法正确")
    except SyntaxError as e:
        print(f"❌ train.py 有语法错误: {e}")
        return False
    
    print()
    return True


def main():
    """主检查函数"""
    print("\n" + "=" * 60)
    print("🔍 运行前环境检查")
    print("=" * 60 + "\n")
    
    checks = [
        ("Python 包", check_imports),
        ("CUDA", check_cuda),
        ("配置文件", check_config),
        ("数据目录", check_data_dirs),
        ("数据样本", check_data_samples),
        ("训练脚本", check_train_script),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} 检查时出错: {e}")
            results.append((name, False))
    
    # 总结
    print("=" * 60)
    print("📊 检查总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r for _, r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有检查通过! 可以开始训练了!")
        print("\n运行以下命令开始训练:")
        print("  python train.py")
    else:
        print("⚠️ 部分检查未通过,请修复上述问题后再训练")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
