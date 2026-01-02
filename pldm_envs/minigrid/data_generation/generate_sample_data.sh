#!/bin/bash
# MiniGridで10エピソードのサンプルデータを生成するスクリプト

# プロジェクトのルートディレクトリ（必要に応じて変更）
PROJECT_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$PROJECT_ROOT"

# 出力パス（必要に応じて変更）
OUTPUT_PATH="pldm_envs/minigrid/data/sample_10episodes"

echo "=========================================="
echo "MiniGrid Sample Data Generation (10 episodes)"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "Output path: $OUTPUT_PATH"
echo ""

# Step 1: プロプリオセプティブデータ（位置、アクションなど）を生成
echo "Step 1/3: Generating proprioceptive data..."
python pldm_envs/minigrid/data_generation/generate_data_from_config.py \
    --config pldm_envs/minigrid/configs/sample_10episodes.yaml \
    --output_path "$OUTPUT_PATH"

if [ $? -ne 0 ]; then
    echo "Error: Data generation failed"
    exit 1
fi

echo ""
echo "Step 1 completed!"
echo ""

# Step 2: PNG画像をレンダリング
echo "Step 2/3: Rendering images..."
python pldm_envs/minigrid/data_generation/render_data.py \
    --data_path "$OUTPUT_PATH"

if [ $? -ne 0 ]; then
    echo "Error: Image rendering failed"
    exit 1
fi

echo ""
echo "Step 2 completed!"
echo ""

# Step 3: 画像をnumpy配列に変換
echo "Step 3/3: Converting images to numpy array..."
python pldm_envs/minigrid/data_generation/postprocess_images.py \
    --data_path "$OUTPUT_PATH"

if [ $? -ne 0 ]; then
    echo "Error: Image postprocessing failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "All steps completed successfully!"
echo "=========================================="
echo "Generated files:"
echo "  - $OUTPUT_PATH/data.p (proprioceptive data)"
echo "  - $OUTPUT_PATH/metadata.pt (config)"
echo "  - $OUTPUT_PATH/images/ (PNG images)"
echo "  - $OUTPUT_PATH/images.zarr (zarr array)"
echo "  - $OUTPUT_PATH/images.npy (numpy array)"
echo ""
echo "You can now use this data for training!"
