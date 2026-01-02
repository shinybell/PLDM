"""
複数のワーカーで生成したデータセットを結合するスクリプト

使用方法:
    # 4つのワーカーで生成したデータを結合
    python pldm_envs/minigrid/data_generation/merge_datasets.py \
        --input_pattern "data/minigrid/worker_*.npz" \
        --output_path data/minigrid/train.npz

    # 特定のファイルを指定して結合
    python pldm_envs/minigrid/data_generation/merge_datasets.py \
        --input_files data/minigrid/worker_0.npz data/minigrid/worker_1.npz \
        --output_path data/minigrid/train.npz
"""

import numpy as np
import argparse
from pathlib import Path
import glob


def load_and_merge_datasets(input_files):
    """
    複数のnpzファイルを読み込んで結合

    Args:
        input_files: 入力ファイルのリスト

    Returns:
        結合されたデータセットの辞書
    """
    all_observations = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_positions = []

    for file_path in sorted(input_files):
        print(f"Loading {file_path}...")
        data = np.load(file_path, allow_pickle=True)

        all_observations.append(data["observations"])
        all_actions.append(data["actions"])
        all_rewards.append(data["rewards"])
        all_dones.append(data["dones"])
        all_positions.append(data["positions"])

    # 結合
    print("\nMerging datasets...")
    merged_data = {
        "observations": np.concatenate(all_observations, axis=0),
        "actions": np.concatenate(all_actions, axis=0),
        "rewards": np.concatenate(all_rewards, axis=0),
        "dones": np.concatenate(all_dones, axis=0),
        "positions": np.concatenate(all_positions, axis=0),
    }

    return merged_data


def main():
    parser = argparse.ArgumentParser(description="複数のデータセットを結合")
    parser.add_argument(
        "--input_pattern",
        type=str,
        default=None,
        help="入力ファイルのパターン (例: 'data/worker_*.npz')",
    )
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        default=None,
        help="入力ファイルのリスト",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="出力ファイルパス (.npz)"
    )

    args = parser.parse_args()

    # 入力ファイルを取得
    if args.input_pattern is not None:
        input_files = glob.glob(args.input_pattern)
        if len(input_files) == 0:
            raise ValueError(f"No files found matching pattern: {args.input_pattern}")
    elif args.input_files is not None:
        input_files = args.input_files
    else:
        raise ValueError("Either --input_pattern or --input_files must be specified")

    print("=" * 60)
    print("Dataset Merging")
    print("=" * 60)
    print(f"Input files ({len(input_files)}):")
    for f in sorted(input_files):
        print(f"  - {f}")
    print(f"Output: {args.output_path}")
    print("=" * 60)

    # データセットを読み込んで結合
    merged_data = load_and_merge_datasets(input_files)

    # 統計情報
    print("\n" + "=" * 60)
    print("Merged Dataset Statistics:")
    print("=" * 60)
    print(f"  Total episodes: {len(merged_data['observations'])}")
    print(f"  Observations shape: {merged_data['observations'].shape}")
    print(f"  Actions shape: {merged_data['actions'].shape}")
    print(f"  Rewards shape: {merged_data['rewards'].shape}")
    print(f"  Dones shape: {merged_data['dones'].shape}")
    print(f"  Positions shape: {merged_data['positions'].shape}")

    # 保存
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving merged dataset to {args.output_path}...")
    np.savez_compressed(args.output_path, **merged_data)

    # ファイルサイズ
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("Dataset merging completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
