#!/bin/bash

# ===================================================================
#                      SBATCH 參數設定 (GPU 版本)
# ===================================================================
# 【修改】輸出日誌檔名
#SBATCH -o diffusion_evaluate_gpu.%j.out
# 【修改】使用 GPU 分區 (GPU40G 通常足夠)
#SBATCH -p GPU80G
#SBATCH --qos=low
# 【修改】作業名稱
#SBATCH -J DiffusionEvalGPU
# 【修改】申請GPU資源
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# ===================================================================
#                      執行環境與命令
# ===================================================================

# --- 1. 初始化計算節點的 Shell 環境 ---
echo "步驟 0: 初始化計算節點 Shell 環境..."
source /profile
echo "------------------------------------------------------"

# --- 2. 打印作業除錯資訊 ---
echo "作業 ID (SLURM_JOB_ID): $SLURM_JOB_ID"
echo "運行節點 (SLURM_NODELIST): $SLURM_NODELIST"
echo "開始時間: $(date)"
echo "------------------------------------------------------"
nvidia-smi
echo "------------------------------------------------------"

# --- 3. 定義並設定工作目錄 ---
PROJECT_ROOT="/lustre/home/2400011491/work/ai_branched_flow/diffusion_force_field"
SCRIPT_PATH="$PROJECT_ROOT/evaluate.py"
cd $PROJECT_ROOT
echo "目前工作目錄: $(pwd)"
echo "------------------------------------------------------"

# --- 4. 載入所需環境模組 ---
echo "步驟 1: 載入 CUDA 模組..."
module load cuda/12.6.0
module list
echo "------------------------------------------------------"

# --- 5. 初始化並啟用您個人的 Conda 環境 ---
echo "步驟 2: 初始化並啟用您個人的 Conda 環境..."
source /lustre/home/2400011491/software/miniconda3/etc/profile.d/conda.sh
conda activate ai_bf
echo "Conda 環境 'ai_bf' 已啟用。"
echo "------------------------------------------------------"

# --- 6. 執行 Python 評估腳本 ---
echo "步驟 3: 開始執行 Python 評估腳本 (GPU加速版)..."
python -u $SCRIPT_PATH
echo "------------------------------------------------------"
echo "評估腳本執行完畢。結束時間: $(date)"