#!/bin/bash
#SBATCH -p gpu_dia
#SBATCH -t 0-12:00
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH -J cell_ft_tune
#SBATCH -o logs/cell_ft_tune_%j.log
#SBATCH -e logs/cell_ft_tune_%j.log

source ~/.bashrc
conda activate openground
set -euo pipefail

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="outputs/auto_runs/cell_ft_tune"
mkdir -p "$OUTPUT_DIR"

PORT=$((29500 + RANDOM % 1000))
GPUS=2

CMD=(torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS} --master_port=${PORT} \
  main.py \
  --config_file config/cfg_odvg.py \
  --datasets config/datasets_cell_debug.json \
  --output_dir "$OUTPUT_DIR" \
  --pretrain_model_path weights/groundingdino_swint_ogc.pth \
  --note cell_ft_tune \
  --amp \
  --options epochs=30 batch_size=6 lr=4e-05 lr_backbone=4e-06 warmup_epochs=2 freeze_keywords=[] )

if [ -f "$OUTPUT_DIR/checkpoint.pth" ]; then
  CMD+=(--resume "$OUTPUT_DIR/checkpoint.pth")
fi

echo "Running: ${CMD[@]}"
"${CMD[@]}"
