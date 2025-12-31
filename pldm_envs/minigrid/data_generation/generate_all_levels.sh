#!/bin/bash
# MiniGrid LongHorizon Level 1-3のデータを生成するスクリプト

# デフォルト設定
N_EPISODES=1000
OBS_SIZE=64
SEED=42
DATA_DIR="data/minigrid"

# 引数パース
while [[ $# -gt 0 ]]; do
  case $1 in
    --n_episodes)
      N_EPISODES="$2"
      shift 2
      ;;
    --obs_size)
      OBS_SIZE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=========================================="
echo "MiniGrid Data Generation - All Levels"
echo "=========================================="
echo "Episodes per level: $N_EPISODES"
echo "Observation size: ${OBS_SIZE}x${OBS_SIZE}"
echo "Seed: $SEED"
echo "Data directory: $DATA_DIR"
echo "=========================================="

# データディレクトリ作成
mkdir -p "$DATA_DIR"

# Level 1
echo ""
echo "Generating Level 1 data..."
python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes "$N_EPISODES" \
  --obs_size "$OBS_SIZE" \
  --seed "$SEED" \
  --output_path "$DATA_DIR/level1_${OBS_SIZE}x${OBS_SIZE}_train.npz"

# Level 2
echo ""
echo "Generating Level 2 data..."
python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level2-v0 \
  --n_episodes "$N_EPISODES" \
  --obs_size "$OBS_SIZE" \
  --seed "$SEED" \
  --output_path "$DATA_DIR/level2_${OBS_SIZE}x${OBS_SIZE}_train.npz"

# Level 3
echo ""
echo "Generating Level 3 data..."
python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level3-v0 \
  --n_episodes "$N_EPISODES" \
  --obs_size "$OBS_SIZE" \
  --seed "$SEED" \
  --output_path "$DATA_DIR/level3_${OBS_SIZE}x${OBS_SIZE}_train.npz"

echo ""
echo "=========================================="
echo "All levels completed!"
echo "=========================================="
echo "Generated files:"
ls -lh "$DATA_DIR"/level*_${OBS_SIZE}x${OBS_SIZE}_train.npz
echo "=========================================="
