"""
MiniGridエピソード全体を可視化するスクリプト

使用方法:
    # 1エピソードを可視化
    python pldm_envs/minigrid/visualize_episode.py \
        --data_path data/minigrid/empty_8x8_debug.npz \
        --episode_idx 0 \
        --output_path episode_0_visualization.png

    # 複数エピソードを可視化
    python pldm_envs/minigrid/visualize_episode.py \
        --data_path data/minigrid/empty_8x8_debug.npz \
        --multiple \
        --n_episodes 5 \
        --output_dir visualizations/

    # 環境を動的に実行して可視化
    python pldm_envs/minigrid/visualize_episode.py \
        --live \
        --env_name MiniGrid-Empty-8x8-v0 \
        --max_steps 200 \
        --output_path live_episode.png
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


# MiniGridアクション名のマッピング
ACTION_NAMES = {
    0: "Turn Left",
    1: "Turn Right",
    2: "Move Forward",
    3: "Pick Up",
    4: "Drop",
    5: "Toggle",
    6: "Done",
}


def visualize_episode(
    data_path: str,
    episode_idx: int = 0,
    output_path: str = None,
    max_frames: int = None
):
    """
    1エピソード全体を可視化

    Args:
        data_path: NPZファイルのパス
        episode_idx: 可視化するエピソードのインデックス
        output_path: 出力画像のパス
        max_frames: 表示する最大フレーム数（Noneの場合は全フレーム）
    """
    print("=" * 60)
    print("MiniGrid Episode Visualization")
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
        raise ValueError(
            f"Episode {episode_idx} does not exist. Total episodes: {len(observations)}"
        )

    episode_obs = observations[episode_idx]
    episode_actions = actions[episode_idx]
    episode_rewards = rewards[episode_idx]
    episode_dones = dones[episode_idx]

    # object型の場合はuint8に変換
    if episode_obs.dtype == object:
        episode_obs = np.array(episode_obs, dtype=np.uint8)
    if episode_actions.dtype == object:
        episode_actions = np.array(episode_actions, dtype=np.int64)
    if episode_rewards.dtype == object:
        episode_rewards = np.array(episode_rewards, dtype=np.float32)
    if episode_dones.dtype == object:
        episode_dones = np.array(episode_dones, dtype=bool)

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

    print(f"  Actual episode length: {actual_length} frames")
    print(f"  Total reward: {episode_rewards[:actual_length].sum():.3f}")

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

        # MiniGridはRGB画像
        ax.imshow(img)

        # タイトル（フレーム番号、アクション、報酬）
        if frame_idx < len(episode_actions):
            action = episode_actions[frame_idx]
            action_name = ACTION_NAMES.get(action, f"Unknown({action})")
            reward = episode_rewards[frame_idx]
            ax.set_title(
                f"t={frame_idx}\n{action_name}\nr={reward:.2f}",
                fontsize=7
            )
        else:
            ax.set_title(f"t={frame_idx}", fontsize=7)

        ax.axis('off')

    # 余ったサブプロットを非表示
    for idx in range(n_frames, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.suptitle(
        f"MiniGrid Episode {episode_idx} - Total Steps: {actual_length}",
        fontsize=14,
        y=0.995
    )
    plt.tight_layout()

    # 保存
    if output_path is None:
        output_path = f"minigrid_episode_{episode_idx}_visualization.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")

    # 統計情報を表示
    print("\n" + "=" * 60)
    print("Episode Statistics:")
    print("=" * 60)
    print(f"  Total frames: {actual_length}")
    print(f"  Total reward: {episode_rewards[:actual_length].sum():.3f}")
    print(f"  Average reward per step: {episode_rewards[:actual_length].mean():.3f}")
    print(f"  Max reward: {episode_rewards[:actual_length].max():.3f}")
    print(f"  Min reward: {episode_rewards[:actual_length].min():.3f}")

    # アクションの分布
    print(f"\nAction distribution:")
    unique_actions, counts = np.unique(
        episode_actions[:actual_length], return_counts=True
    )
    for action, count in zip(unique_actions, counts):
        percentage = (count / actual_length) * 100
        action_name = ACTION_NAMES.get(action, f"Unknown({action})")
        print(f"  {action_name} (#{action}): {count} times ({percentage:.1f}%)")

    print("\n" + "=" * 60)
    plt.close()


def visualize_multiple_episodes(
    data_path: str, n_episodes: int = 5, output_dir: str = "episode_visualizations"
):
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
        visualize_episode(
            data_path, episode_idx=i, output_path=str(output_path), max_frames=100
        )
        print()


def visualize_live_episode(
    env_name: str = "MiniGrid-Empty-8x8-v0",
    max_steps: int = 200,
    output_path: str = None,
    seed: int = 42,
):
    """
    環境を実行してライブで可視化

    Args:
        env_name: MiniGrid環境名
        max_steps: 最大ステップ数
        output_path: 出力画像のパス
        seed: ランダムシード
    """
    print("=" * 60)
    print("MiniGrid Live Episode Visualization")
    print("=" * 60)

    try:
        import gymnasium as gym
        import minigrid
        from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

        gym.register_envs(minigrid)
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install: pip install minigrid")
        return

    # 環境の作成
    env = gym.make(env_name, max_steps=max_steps, tile_size=8, render_mode=None, highlight=False)
    env = RGBImgObsWrapper(env, tile_size=8)  # 完全観測のRGB画像に変換
    env = ImgObsWrapper(env)  # 辞書から画像のみを取り出す

    print(f"\nRunning episode in {env_name}...")
    print(f"  Max steps: {max_steps}")
    print(f"  Seed: {seed}")

    # エピソード実行
    obs_list = []
    action_list = []
    reward_list = []

    obs, info = env.reset(seed=seed)
    obs_list.append(obs)

    total_reward = 0
    for step in range(max_steps):
        action = env.action_space.sample()  # ランダムアクション
        obs, reward, terminated, truncated, info = env.step(action)

        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(reward)
        total_reward += reward

        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}")
            break

    env.close()

    print(f"  Total steps: {len(action_list)}")
    print(f"  Total reward: {total_reward:.3f}")

    # 可視化
    n_frames = len(obs_list)
    n_cols = min(10, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols

    print(f"\nCreating visualization grid: {n_rows}x{n_cols}")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(n_frames):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        ax.imshow(obs_list[idx])

        if idx < len(action_list):
            action = action_list[idx]
            action_name = ACTION_NAMES.get(action, f"Unknown({action})")
            reward = reward_list[idx]
            ax.set_title(f"t={idx}\n{action_name}\nr={reward:.2f}", fontsize=7)
        else:
            ax.set_title(f"t={idx} (initial)", fontsize=7)

        ax.axis('off')

    # 余ったサブプロットを非表示
    for idx in range(n_frames, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.suptitle(
        f"{env_name} - Live Episode - Total Steps: {len(action_list)}",
        fontsize=14,
        y=0.995
    )
    plt.tight_layout()

    # 保存
    if output_path is None:
        output_path = f"minigrid_live_episode.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")

    # アクション分布
    print("\nAction distribution:")
    unique_actions, counts = np.unique(action_list, return_counts=True)
    for action, count in zip(unique_actions, counts):
        percentage = (count / len(action_list)) * 100
        action_name = ACTION_NAMES.get(action, f"Unknown({action})")
        print(f"  {action_name} (#{action}): {count} times ({percentage:.1f}%)")

    print("\n" + "=" * 60)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniGridエピソード可視化")
    parser.add_argument("--data_path", type=str, help="NPZファイルのパス")
    parser.add_argument(
        "--episode_idx", type=int, default=0, help="可視化するエピソードのインデックス"
    )
    parser.add_argument("--output_path", type=str, default=None, help="出力画像のパス")
    parser.add_argument(
        "--max_frames", type=int, default=None, help="表示する最大フレーム数"
    )
    parser.add_argument("--multiple", action="store_true", help="複数エピソードを可視化")
    parser.add_argument(
        "--n_episodes", type=int, default=5, help="可視化するエピソード数（--multiple使用時）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="episode_visualizations",
        help="出力ディレクトリ（--multiple使用時）",
    )
    parser.add_argument("--live", action="store_true", help="ライブで環境を実行して可視化")
    parser.add_argument(
        "--env_name",
        type=str,
        default="MiniGrid-Empty-8x8-v0",
        help="MiniGrid環境名（--live使用時）",
    )
    parser.add_argument(
        "--max_steps", type=int, default=200, help="最大ステップ数（--live使用時）"
    )
    parser.add_argument("--seed", type=int, default=42, help="ランダムシード（--live使用時）")

    args = parser.parse_args()

    if args.live:
        visualize_live_episode(args.env_name, args.max_steps, args.output_path, args.seed)
    elif args.multiple:
        if args.data_path is None:
            print("Error: --data_path is required for --multiple mode")
            exit(1)
        visualize_multiple_episodes(args.data_path, args.n_episodes, args.output_dir)
    else:
        if args.data_path is None:
            print("Error: --data_path is required (or use --live mode)")
            exit(1)

        # output_pathが指定されていない場合、output_dirを使用
        output_path = args.output_path
        if output_path is None and args.output_dir:
            output_path = f"{args.output_dir}/minigrid_episode_{args.episode_idx}_visualization.png"

        visualize_episode(args.data_path, args.episode_idx, output_path, args.max_frames)
