#!/bin/bash

# ===================================================================
#                      SBATCH 參數設定
# ===================================================================
# 【修改】作業輸出日誌檔名
#SBATCH -o unet_train.%j.out
# 【保留】指定運行分區 (Partition)
#SBATCH -p GPU40G
# 【保留】指定服務品質 (Quality of Service)
#SBATCH --qos=low
# 【修改】作業名稱
#SBATCH -J UNetTrain
# 【保留】節點、GPU、任務和CPU核心數配置
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# ===================================================================
#                      執行環境與命令
# ===================================================================

# --- 1. 初始化計算節點的 Shell 環境 ---
echo "步驟 0: 初始化計算節點 Shell 環境..."
source /etc/profile
echo "Shell 環境已初始化。"
echo "------------------------------------------------------"

# --- 2. 打印作業除錯資訊 ---
echo "作業 ID (SLURM_JOB_ID): $SLURM_JOB_ID"
echo "運行節點 (SLURM_NODELIST): $SLURM_NODELIST"
echo "開始時間: $(date)"
echo "------------------------------------------------------"
nvidia-smi
echo "------------------------------------------------------"

# --- 3. 定義並設定工作目錄 ---
# 【修改】將 PROJECT_ROOT 指向您的 unet_force_field_predictor 專案
PROJECT_ROOT="/lustre/home/2400011491/work/ai_branched_flow/unet_force_field_predictor"
# 【無需修改】訓練腳本路徑會自動更新
SCRIPT_PATH="$PROJECT_ROOT/train.py"

cd $PROJECT_ROOT
echo "目前工作目錄: $(pwd)"
echo "------------------------------------------------------"

# --- 4. 載入所需環境模組 ---
echo "步驟 1: 載入 CUDA 模組..."
module load cuda/12.6.0
echo "CUDA 模組已載入。"
module list
echo "------------------------------------------------------"

# --- 5. 初始化並啟用您個人的 Conda 環境 ---
echo "步驟 2: 初始化並啟用您個人的 Conda 環境 'ai_bf'..."
# 【保留】這會初始化您自己安裝的 Conda
source /lustre/home/2400011491/software/miniconda3/etc/profile.d/conda.sh
# 【保留】這會啟用您的目標 Conda 環境
conda activate ai_bf
echo "Conda 環境 'ai_bf' 已啟用。"
echo "Python 執行檔路徑: $(which python)"
echo "------------------------------------------------------"

# --- 6. 執行 Python 訓練腳本 ---
echo "步驟 3: 開始執行訓練腳本 $SCRIPT_PATH..."
python $SCRIPT_PATH
echo "------------------------------------------------------"
echo "訓練腳本執行完畢。"
echo "結束時間: $(date)"
echo "作業完成。"