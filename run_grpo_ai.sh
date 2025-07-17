#!/bin/bash
#SBATCH --job-name=trl_grpo_test       # 作业名
#SBATCH --partition=ai                 # AI/GPU 分区（48 hr 最大）
#SBATCH --account=ai240246-ai          # 你的 AI Service Unit 账目
#SBATCH --nodes=1                      # 单节点
#SBATCH --ntasks=1                     # 串行任务
#SBATCH --gres=gpu:2                   # 请求 2 张 GPU
#SBATCH --cpus-per-task=8              # 每任务 8 个 CPU 核
#SBATCH --mem=128G                      # 128 GB 内存
#SBATCH --time=01:00:00                # 最长运行 1 小时（可调）
#SBATCH --output=logs/grpo.%j.out      # 标准输出日志
#SBATCH --error=logs/grpo.%j.err       # 标准错误日志

# ——— 清理并加载环境 ———
module purge
module load modtree/gpu               # 自动带 cuda/11.2.2、GCC、MPI 等
module list

nvidia-smi

# 加载 Conda
source ~/chaohan/anaconda3/etc/profile.d/conda.sh
conda activate trl_env

export HF_HOME="$HOME/chaohan/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_TRANSFORMERS_CACHE="$HF_HOME"        # v5 以后只认 HF_HOME
export HF_HUB_CACHE="$HF_HOME"

# 进入项目目录
cd ~/chaohan/projects/trl

# 确保日志和输出目录存在
mkdir -p logs grpo_test_out

# 运行你的测试脚本
python -m torch.distributed.run --nproc_per_node=2 test_grpo.py
