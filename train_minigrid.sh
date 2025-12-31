#!/bin/bash
# MiniGrid Level1でPLDMを訓練するスクリプト

set -e

# 仮想環境をアクティベート
source .venv/bin/activate

# PYTHONPATHを設定
export PYTHONPATH=/Users/shunsei/works/MatsuoLab/WorldModel/PLDM:$PYTHONPATH

# 設定ファイル
CONFIG="pldm/configs/minigrid/level1_test.yaml"

# データファイルの確認
TRAIN_DATA="data/minigrid/level1_train_small.npz"
VAL_DATA="data/minigrid/level1_val_small.npz"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found at $TRAIN_DATA"
    echo "Please generate data first using:"
    echo "  python pldm_envs/minigrid/data_generation/generate_pldm_data.py \\"
    echo "    --env_name MiniGrid-LongHorizon-Level1-v0 \\"
    echo "    --n_episodes 50 \\"
    echo "    --output_path $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "⚠️  Warning: Validation data not found at $VAL_DATA"
    echo "Continuing without validation data..."
fi

echo "============================================================"
echo "PLDM MiniGrid Training"
echo "============================================================"
echo "Config: $CONFIG"
echo "Train data: $TRAIN_DATA"
echo "Val data: $VAL_DATA"
echo "============================================================"
echo ""

# 訓練実行
python pldm/train.py \
    --config-name level1_test \
    --config-path configs/minigrid

echo ""
echo "============================================================"
echo "Training completed!"
echo "============================================================"
