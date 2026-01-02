#!/bin/bash
# 並列データ生成スクリプト
#
# 使用方法:
#   # venv環境で実行
#   source .venv/bin/activate
#   bash pldm_envs/minigrid/data_generation/parallel_generate.sh \
#       --env_name MiniGrid-Empty-8x8-v0 \
#       --n_episodes 10000 \
#       --workers 4 \
#       --output_dir data/minigrid/parallel_test
#
# オプション:
#   --env_name: MiniGrid環境名 (デフォルト: MiniGrid-Empty-8x8-v0)
#   --n_episodes: 総エピソード数 (デフォルト: 10000)
#   --workers: ワーカー数 (デフォルト: 4)
#   --output_dir: 出力ディレクトリ (デフォルト: data/minigrid/parallel)
#   --max_steps: 最大ステップ数 (オプション)
#   --resize: リサイズサイズ (オプション、例: 72)
#   --pad_length: パディング長 (オプション)

# デフォルト値
ENV_NAME="MiniGrid-Empty-8x8-v0"
N_EPISODES=10000
WORKERS=4
OUTPUT_DIR="data/minigrid/parallel"
MAX_STEPS=""
RESIZE=""
PAD_LENGTH=""

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --env_name)
            ENV_NAME="$2"
            shift 2
            ;;
        --n_episodes)
            N_EPISODES="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="--max_steps $2"
            shift 2
            ;;
        --resize)
            RESIZE="--resize $2"
            shift 2
            ;;
        --pad_length)
            PAD_LENGTH="--pad_length $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# エピソード数がワーカー数で割り切れるか確認
if [ $((N_EPISODES % WORKERS)) -ne 0 ]; then
    echo "Error: n_episodes ($N_EPISODES) must be divisible by workers ($WORKERS)"
    exit 1
fi

echo "========================================"
echo "Parallel Data Generation"
echo "========================================"
echo "Environment: $ENV_NAME"
echo "Total episodes: $N_EPISODES"
echo "Workers: $WORKERS"
echo "Episodes per worker: $((N_EPISODES / WORKERS))"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

# 出力ディレクトリを作成
mkdir -p "$OUTPUT_DIR"

# PYTHONPATHを設定
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 各ワーカーをバックグラウンドで起動
PIDS=()
for ((i=0; i<WORKERS; i++)); do
    OUTPUT_FILE="$OUTPUT_DIR/worker_$i.npz"

    echo "Starting worker $i -> $OUTPUT_FILE"

    python pldm_envs/minigrid/data_generation/generate_data.py \
        --env_name "$ENV_NAME" \
        --n_episodes "$N_EPISODES" \
        --output_path "$OUTPUT_FILE" \
        --workers_num "$WORKERS" \
        --worker_id "$i" \
        $MAX_STEPS \
        $RESIZE \
        $PAD_LENGTH \
        > "$OUTPUT_DIR/worker_$i.log" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All workers started. Waiting for completion..."
echo "PIDs: ${PIDS[@]}"
echo ""
echo "Monitor progress with:"
echo "  tail -f $OUTPUT_DIR/worker_*.log"
echo ""

# すべてのワーカーの完了を待つ
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    echo "Waiting for worker $i (PID: $PID)..."

    if wait $PID; then
        echo "  Worker $i completed successfully"
    else
        echo "  Worker $i failed!"
        FAILED=1
    fi
done

if [ $FAILED -eq 1 ]; then
    echo ""
    echo "========================================"
    echo "ERROR: Some workers failed"
    echo "========================================"
    echo "Check logs in $OUTPUT_DIR/"
    exit 1
fi

echo ""
echo "========================================"
echo "All workers completed successfully!"
echo "========================================"
echo ""
echo "Merging datasets..."

# データセットを結合
python pldm_envs/minigrid/data_generation/merge_datasets.py \
    --input_pattern "$OUTPUT_DIR/worker_*.npz" \
    --output_path "$OUTPUT_DIR/train.npz"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Success!"
    echo "========================================"
    echo "Output file: $OUTPUT_DIR/train.npz"
    echo ""
    echo "Cleanup temporary worker files? (y/n)"
    read -r CLEANUP

    if [ "$CLEANUP" = "y" ] || [ "$CLEANUP" = "Y" ]; then
        echo "Removing worker files..."
        rm "$OUTPUT_DIR"/worker_*.npz
        rm "$OUTPUT_DIR"/worker_*.log
        echo "Cleanup completed"
    fi
else
    echo ""
    echo "========================================"
    echo "ERROR: Dataset merging failed"
    echo "========================================"
    exit 1
fi
