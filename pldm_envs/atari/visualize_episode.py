"""
1エピソード全体を可視化するスクリプト

使用方法:
    python pldm_envs/atari/visualize_episode.py \
        --data_path data/atari/pacman_debug.npz \
        --episode_idx 0 \
        --output_path episode_0_visualization.png
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def visualize_episode(data_path: str, episode_idx: int = 0, output_path: str = None, max_frames: int = None):
    """
    1エピソード全体を可視化

    Args:
        data_path: NPZファイルのパス
        episode_idx: 可視化するエピソードのインデックス
        output_path: 出力画像のパス
        max_frames: 表示する最大フレーム数（Noneの場合は全フレーム）
    """
    print("=" * 60)
    print("Atari Episode Visualization")
    print("=" * 60)

    # データ読み込み
    print(f"\nLoading data from: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    observations = data['observations']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']

    print(f"Data shapes:")
    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Dones: {dones.shape}")

    # 指定エピソードを取得
    if episode_idx >= len(observations):
        raise ValueError(f"Episode {episode_idx} does not exist. Total episodes: {len(observations)}")

    episode_obs = observations[episode_idx]
    episode_actions = actions[episode_idx]
    episode_rewards = rewards[episode_idx]
    episode_dones = dones[episode_idx]

    print(f"\nEpisode {episode_idx}:")
    print(f"  Observations: {episode_obs.shape}")
    print(f"  Total frames: {len(episode_obs)}")

    # 実際にゲームが進行していた部分を特定（doneがTrueになるまで）
    if len(episode_dones) > 0:
        done_indices = np.where(episode_dones)[0]
        if len(done_indices) > 0:
            actual_length = done_indices[0] + 1  # 最初のdoneまで
        else:
            actual_length = len(episode_obs)
    else:
        actual_length = len(episode_obs)

    print(f"  Actual game length: {actual_length} frames")
    print(f"  Total reward: {episode_rewards[:actual_length].sum():.1f}")

    # 表示するフレーム数を決定
    if max_frames is not None:
        display_length = min(actual_length, max_frames)
    else:
        display_length = actual_length

    # フレームをサンプリング（多すぎる場合）
    if display_length > 100:
        # 100フレームに間引く
        indices = np.linspace(0, display_length - 1, 100, dtype=int)
        print(f"  Sampling {len(indices)} frames from {display_length} frames")
    else:
        indices = np.arange(display_length)

    # グリッドサイズを計算
    n_frames = len(indices)
    n_cols = min(10, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols

    print(f"\nCreating visualization grid: {n_rows}x{n_cols}")

    # 可視化
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, frame_idx in enumerate(indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # 画像を表示
        img = episode_obs[frame_idx]

        # グレースケールの場合
        if len(img.shape) == 2 or img.shape[2] == 1:
            if len(img.shape) == 3:
                img = img[:, :, 0]
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

        # タイトル（フレーム番号、アクション、報酬）
        if frame_idx < len(episode_actions):
            action = episode_actions[frame_idx]
            reward = episode_rewards[frame_idx]
            ax.set_title(f"t={frame_idx}\na={action}, r={reward:.1f}", fontsize=8)
        else:
            ax.set_title(f"t={frame_idx}", fontsize=8)

        ax.axis('off')

    # 余ったサブプロットを非表示
    for idx in range(n_frames, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    # 保存
    if output_path is None:
        output_path = f"episode_{episode_idx}_visualization.png"

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")

    # 統計情報を表示
    print("\n" + "=" * 60)
    print("Episode Statistics:")
    print("=" * 60)
    print(f"  Total frames: {actual_length}")
    print(f"  Total reward: {episode_rewards[:actual_length].sum():.1f}")
    print(f"  Average reward per step: {episode_rewards[:actual_length].mean():.3f}")
    print(f"  Max reward: {episode_rewards[:actual_length].max():.1f}")
    print(f"  Min reward: {episode_rewards[:actual_length].min():.1f}")

    # アクションの分布
    print(f"\nAction distribution:")
    unique_actions, counts = np.unique(episode_actions[:actual_length], return_counts=True)
    for action, count in zip(unique_actions, counts):
        percentage = (count / actual_length) * 100
        print(f"  Action {action}: {count} times ({percentage:.1f}%)")

    print("\n" + "=" * 60)
    plt.close()


def visualize_multiple_episodes(data_path: str, n_episodes: int = 5, output_dir: str = "episode_visualizations"):
    """
    複数エピソードを可視化

    Args:
        data_path: NPZファイルのパス
        n_episodes: 可視化するエピソード数
        output_dir: 出力ディレクトリ
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_path, allow_pickle=True)
    total_episodes = len(data['observations'])
    n_episodes = min(n_episodes, total_episodes)

    print(f"Visualizing {n_episodes} episodes...")

    for i in range(n_episodes):
        output_path = output_dir / f"episode_{i}.png"
        visualize_episode(data_path, episode_idx=i, output_path=str(output_path), max_frames=100)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atariエピソード可視化")
    parser.add_argument("--data_path", type=str, required=True, help="NPZファイルのパス")
    parser.add_argument("--episode_idx", type=int, default=0, help="可視化するエピソードのインデックス")
    parser.add_argument("--output_path", type=str, default=None, help="出力画像のパス")
    parser.add_argument("--max_frames", type=int, default=None, help="表示する最大フレーム数")
    parser.add_argument("--multiple", action="store_true", help="複数エピソードを可視化")
    parser.add_argument("--n_episodes", type=int, default=5, help="可視化するエピソード数（--multiple使用時）")
    parser.add_argument("--output_dir", type=str, default="episode_visualizations", help="出力ディレクトリ（--multiple使用時）")

    args = parser.parse_args()

    if args.multiple:
        visualize_multiple_episodes(args.data_path, args.n_episodes, args.output_dir)
    else:
        visualize_episode(args.data_path, args.episode_idx, args.output_path, args.max_frames)
