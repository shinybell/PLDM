#!/bin/bash
# Long-Horizon環境のテストデータ生成と可視化を自動化するスクリプト
#
# 使用方法:
#   bash pldm_envs/minigrid/test_all_levels.sh
#
# オプション:
#   --n-episodes N     各レベルで生成するエピソード数 (デフォルト: 10)
#   --only-level N     指定したレベルのみ実行 (例: --only-level 1)
#   --visualize-only   データ生成をスキップして可視化のみ実行

set -e  # エラーが発生したら即座に終了

# デフォルト設定
N_EPISODES=10
ONLY_LEVEL=""
VISUALIZE_ONLY=false

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-episodes)
            N_EPISODES="$2"
            shift 2
            ;;
        --only-level)
            ONLY_LEVEL="$2"
            shift 2
            ;;
        --visualize-only)
            VISUALIZE_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# プロジェクトルート設定
PROJECT_ROOT="/Users/shunsei/works/MatsuoLab/WorldModel/PLDM"
cd "$PROJECT_ROOT"

# 仮想環境のアクティベート
source .venv/bin/activate

# PYTHONPATHの設定
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "========================================="
echo "Long-Horizon Environments Test Suite"
echo "========================================="
echo "Episodes per level: $N_EPISODES"
echo "Visualize only: $VISUALIZE_ONLY"
if [ -n "$ONLY_LEVEL" ]; then
    echo "Only level: $ONLY_LEVEL"
fi
echo "========================================="
echo ""

# ディレクトリ作成
mkdir -p data/minigrid
mkdir -p visualizations/minigrid/level1
mkdir -p visualizations/minigrid/level2
mkdir -p visualizations/minigrid/level3

# Level 1: 24x24 Grid, Simple walls
if [ -z "$ONLY_LEVEL" ] || [ "$ONLY_LEVEL" = "1" ]; then
    echo ""
    echo "========================================="
    echo "Level 1: 24x24 Grid (Simple Walls)"
    echo "========================================="

    if [ "$VISUALIZE_ONLY" = false ]; then
        echo ""
        echo "[1/3] Generating test data..."
        python pldm_envs/minigrid/data_generation/generate_data.py \
            --env_name MiniGrid-LongHorizon-Level1-v0 \
            --n_episodes "$N_EPISODES" \
            --max_steps 256 \
            --resize 64 \
            --seed 42 \
            --output_path data/minigrid/long_horizon_level1_debug.npz
    else
        echo "[1/3] Skipping data generation (visualize-only mode)"
    fi

    echo ""
    echo "[2/3] Visualizing episode 0..."
    python pldm_envs/minigrid/visualize_episode.py \
        --data_path data/minigrid/long_horizon_level1_debug.npz \
        --episode_idx 0 \
        --output_dir visualizations/minigrid/level1 \
        --max_frames 50

    echo ""
    echo "[3/3] Visualizing episode 1..."
    python pldm_envs/minigrid/visualize_episode.py \
        --data_path data/minigrid/long_horizon_level1_debug.npz \
        --episode_idx 1 \
        --output_dir visualizations/minigrid/level1 \
        --max_frames 50

    echo ""
    echo "✓ Level 1 completed!"
    echo "  Data: data/minigrid/long_horizon_level1_debug.npz"
    echo "  Visualizations: visualizations/minigrid/level1/"
else
    echo ""
    echo "========================================="
    echo "Level 1: SKIPPED"
    echo "========================================="
fi

# Level 2: 24x24 Grid, Medium walls
if [ -z "$ONLY_LEVEL" ] || [ "$ONLY_LEVEL" = "2" ]; then
    echo ""
    echo "========================================="
    echo "Level 2: 24x24 Grid (Medium Walls)"
    echo "========================================="

    if [ "$VISUALIZE_ONLY" = false ]; then
        echo ""
        echo "[1/3] Generating test data..."
        python pldm_envs/minigrid/data_generation/generate_data.py \
            --env_name MiniGrid-LongHorizon-Level2-v0 \
            --n_episodes "$N_EPISODES" \
            --max_steps 256 \
            --resize 64 \
            --seed 42 \
            --output_path data/minigrid/long_horizon_level2_debug.npz
    else
        echo "[1/3] Skipping data generation (visualize-only mode)"
    fi

    echo ""
    echo "[2/3] Visualizing episode 0..."
    python pldm_envs/minigrid/visualize_episode.py \
        --data_path data/minigrid/long_horizon_level2_debug.npz \
        --episode_idx 0 \
        --output_dir visualizations/minigrid/level2 \
        --max_frames 50

    echo ""
    echo "[3/3] Visualizing episode 1..."
    python pldm_envs/minigrid/visualize_episode.py \
        --data_path data/minigrid/long_horizon_level2_debug.npz \
        --episode_idx 1 \
        --output_dir visualizations/minigrid/level2 \
        --max_frames 50

    echo ""
    echo "✓ Level 2 completed!"
    echo "  Data: data/minigrid/long_horizon_level2_debug.npz"
    echo "  Visualizations: visualizations/minigrid/level2/"
else
    echo ""
    echo "========================================="
    echo "Level 2: SKIPPED"
    echo "========================================="
fi

# Level 3: 24x24 Grid, Complex walls
if [ -z "$ONLY_LEVEL" ] || [ "$ONLY_LEVEL" = "3" ]; then
    echo ""
    echo "========================================="
    echo "Level 3: 24x24 Grid (Complex Walls)"
    echo "========================================="

    if [ "$VISUALIZE_ONLY" = false ]; then
        echo ""
        echo "[1/3] Generating test data..."
        python pldm_envs/minigrid/data_generation/generate_data.py \
            --env_name MiniGrid-LongHorizon-Level3-v0 \
            --n_episodes "$N_EPISODES" \
            --max_steps 256 \
            --resize 64 \
            --seed 42 \
            --output_path data/minigrid/long_horizon_level3_debug.npz
    else
        echo "[1/3] Skipping data generation (visualize-only mode)"
    fi

    echo ""
    echo "[2/3] Visualizing episode 0..."
    python pldm_envs/minigrid/visualize_episode.py \
        --data_path data/minigrid/long_horizon_level3_debug.npz \
        --episode_idx 0 \
        --output_dir visualizations/minigrid/level3 \
        --max_frames 50

    echo ""
    echo "[3/3] Visualizing episode 1..."
    python pldm_envs/minigrid/visualize_episode.py \
        --data_path data/minigrid/long_horizon_level3_debug.npz \
        --episode_idx 1 \
        --output_dir visualizations/minigrid/level3 \
        --max_frames 50

    echo ""
    echo "✓ Level 3 completed!"
    echo "  Data: data/minigrid/long_horizon_level3_debug.npz"
    echo "  Visualizations: visualizations/minigrid/level3/"
else
    echo ""
    echo "========================================="
    echo "Level 3: SKIPPED"
    echo "========================================="
fi

# 完了メッセージ
echo ""
echo "========================================="
echo "ALL LEVELS COMPLETED!"
echo "========================================="
echo ""
echo "Generated files:"
if [ -z "$ONLY_LEVEL" ] || [ "$ONLY_LEVEL" = "1" ]; then
    echo "  Level 1:"
    echo "    - data/minigrid/long_horizon_level1_debug.npz"
    echo "    - visualizations/minigrid/level1/*.png"
fi
if [ -z "$ONLY_LEVEL" ] || [ "$ONLY_LEVEL" = "2" ]; then
    echo "  Level 2:"
    echo "    - data/minigrid/long_horizon_level2_debug.npz"
    echo "    - visualizations/minigrid/level2/*.png"
fi
if [ -z "$ONLY_LEVEL" ] || [ "$ONLY_LEVEL" = "3" ]; then
    echo "  Level 3:"
    echo "    - data/minigrid/long_horizon_level3_debug.npz"
    echo "    - visualizations/minigrid/level3/*.png"
fi
echo ""
echo "Next steps:"
echo "  1. Review visualizations in visualizations/minigrid/"
echo "  2. Generate training data using commands in QUICKSTART_LONG_HORIZON.md"
echo "  3. Train models using config files in pldm_envs/minigrid/configs/"
echo ""
