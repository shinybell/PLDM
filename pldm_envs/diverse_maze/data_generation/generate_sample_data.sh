#!/bin/bash
# Diverse Mazeで10エピソードのサンプルデータを生成するスクリプト

# プロジェクトのルートディレクトリ（必要に応じて変更）
PROJECT_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$PROJECT_ROOT"

# 出力パス
OUTPUT_PATH="pldm_envs/diverse_maze/data/sample_10episodes"

echo "=========================================="
echo "Diverse Maze Sample Data Generation (10 episodes)"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "Output path: $OUTPUT_PATH"
echo ""

# Step 1: プロプリオセプティブデータ（位置、アクションなど）を生成
echo "Step 1/3: Generating proprioceptive data..."
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/diverse_maze/data_generation/generate_data.py \
    --output_path "$OUTPUT_PATH" \
    --config pldm_envs/diverse_maze/configs/sample_10episodes.yaml

if [ $? -ne 0 ]; then
    echo "Error: Data generation failed"
    exit 1
fi

echo ""
echo "Step 1 completed!"
echo "Generated files:"
echo "  - $OUTPUT_PATH/data.p"
echo "  - $OUTPUT_PATH/metadata.pt"
echo "  - $OUTPUT_PATH/train_maps.pt"
echo ""

# Step 2: PNG画像をレンダリング
echo "Step 2/3: Rendering images..."
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/diverse_maze/data_generation/render_data.py \
    --data_path "$OUTPUT_PATH"

if [ $? -ne 0 ]; then
    echo "Error: Image rendering failed"
    exit 1
fi

echo ""
echo "Step 2 completed!"
echo "Generated: $OUTPUT_PATH/images/*.png"
echo ""

# Step 3: 画像をnumpy配列に変換
echo "Step 3/3: Converting images to numpy array..."
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/diverse_maze/data_generation/postprocess_images.py \
    --data_path "$OUTPUT_PATH"

if [ $? -ne 0 ]; then
    echo "Error: Image postprocessing failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "All steps completed successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - $OUTPUT_PATH/data.p (proprioceptive data)"
echo "  - $OUTPUT_PATH/metadata.pt (config)"
echo "  - $OUTPUT_PATH/train_maps.pt (map layouts)"
echo "  - $OUTPUT_PATH/images/ (PNG images)"
echo "  - $OUTPUT_PATH/images.zarr (zarr array)"
echo "  - $OUTPUT_PATH/images.npy (numpy array) ← USE THIS"
echo ""
echo "Data shape info:"
PYTHONPATH=$PWD:$PYTHONPATH python -c "import numpy as np; arr=np.load('$OUTPUT_PATH/images.npy'); print(f'  Images shape: {arr.shape}'); print(f'  File size: {arr.nbytes / (1024*1024):.2f} MB')" 2>/dev/null || echo "  (Run after completion)"
echo ""
echo "You can now use this data for training!"
