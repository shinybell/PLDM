"""
MiniGrid環境のデータ生成スクリプト

Gymnasiumを使ってMiniGrid環境からデータを収集します。

使用方法:
    # ランダムポリシーでデータ生成 (Empty-8x8, 200ステップ)
    python pldm_envs/minigrid/data_generation/generate_data.py \\
        --env_name MiniGrid-Empty-8x8-v0 \\
        --n_episodes 1000 \\
        --max_steps 200 \\
        --output_path data/minigrid/empty_8x8_train.npz

    # デバッグ用（少量データ）
    python pldm_envs/minigrid/data_generation/generate_data.py \\
        --env_name MiniGrid-Empty-8x8-v0 \\
        --n_episodes 10 \\
        --max_steps 200 \\
        --output_path data/minigrid/empty_8x8_debug.npz

インストール要件:
    pip install minigrid
"""

import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

gym.register_envs(minigrid)

# Register custom long-horizon environments
import pldm_envs.minigrid
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    import warnings

    warnings.warn(
        "opencv-python not found. Resizing will use a slower fallback method."
    )


def generate_episode(env, policy, max_steps=1000, seed=None):
    """
    1エピソードのデータを生成

    Args:
        env: Gymnasium環境
        policy: ポリシー関数 (obs -> action)
        max_steps: 最大ステップ数
        seed: ランダムシード

    Returns:
        dict: エピソードデータ
            - observations: [T, H, W, C]
            - actions: [T-1]
            - rewards: [T-1]
            - dones: [T-1]
            - positions: [T, 2] - エージェント位置 (x, y)
    """
    obs_list = []
    action_list = []
    reward_list = []
    done_list = []
    position_list = []

    obs, info = env.reset(seed=seed)
    obs_list.append(obs)

    # エージェント位置を取得（MiniGrid環境から）
    agent_pos = env.unwrapped.agent_pos
    # agent_posはtupleなのでnumpy配列に変換
    position_list.append(np.array(agent_pos, dtype=np.float32))

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(terminated or truncated)

        # エージェント位置を取得
        agent_pos = env.unwrapped.agent_pos
        position_list.append(np.array(agent_pos, dtype=np.float32))

        if terminated or truncated:
            break

    return {
        "observations": np.array(obs_list),
        "actions": np.array(action_list),
        "rewards": np.array(reward_list),
        "dones": np.array(done_list),
        "positions": np.array(position_list, dtype=np.float32),
    }


def create_random_policy(env):
    """ランダムポリシーを作成"""

    def policy(obs):
        return env.action_space.sample()

    return policy


def resize_observations(observations, target_size):
    """
    観測画像をリサイズ

    Args:
        observations: [T, H, W, C] の観測画像
        target_size: リサイズ後のサイズ (int)

    Returns:
        リサイズされた観測画像 [T, target_size, target_size, C]
    """
    if HAS_CV2:
        # OpenCVを使用（高速）
        resized = []
        for obs in observations:
            resized_obs = cv2.resize(
                obs, (target_size, target_size), interpolation=cv2.INTER_AREA
            )
            # グレースケールの場合は次元を追加
            if len(resized_obs.shape) == 2:
                resized_obs = resized_obs[:, :, np.newaxis]
            resized.append(resized_obs)
        return np.array(resized)
    else:
        # NumPyのみを使用（遅い）
        import warnings

        warnings.warn(
            "Using slow fallback for resizing. Install opencv-python for better performance."
        )

        T, H, W, C = observations.shape
        resized = np.zeros((T, target_size, target_size, C), dtype=observations.dtype)

        for t in range(T):
            for c in range(C):
                # 簡易的なリサイズ（最近傍補間）
                y_ratio = H / target_size
                x_ratio = W / target_size

                for i in range(target_size):
                    for j in range(target_size):
                        src_y = int(i * y_ratio)
                        src_x = int(j * x_ratio)
                        resized[t, i, j, c] = observations[t, src_y, src_x, c]

        return resized


def pad_episode(episode, target_length):
    """
    エピソードを指定長にパディング

    Args:
        episode: エピソードデータ
        target_length: 目標長

    Returns:
        パディングされたエピソード
    """
    current_length = len(episode["observations"])

    if current_length >= target_length:
        # 切り詰め
        return {
            "observations": episode["observations"][:target_length],
            "actions": episode["actions"][: target_length - 1],
            "rewards": episode["rewards"][: target_length - 1],
            "dones": episode["dones"][: target_length - 1],
        }
    else:
        # パディング
        pad_length = target_length - current_length

        # 最後のフレームを繰り返してパディング
        last_obs = episode["observations"][-1]
        padded_obs = np.concatenate(
            [
                episode["observations"],
                np.repeat(last_obs[np.newaxis], pad_length, axis=0),
            ],
            axis=0,
        )

        # アクション、報酬、doneは0でパディング
        padded_actions = np.concatenate(
            [episode["actions"], np.zeros(pad_length, dtype=episode["actions"].dtype)],
            axis=0,
        )

        padded_rewards = np.concatenate(
            [episode["rewards"], np.zeros(pad_length, dtype=episode["rewards"].dtype)],
            axis=0,
        )

        padded_dones = np.concatenate(
            [
                episode["dones"],
                np.ones(pad_length, dtype=bool),  # パディング部分は終了扱い
            ],
            axis=0,
        )

        padded_positions = np.concatenate(
            [
                episode["positions"],
                np.tile(episode["positions"][-1:], (pad_length, 1)),  # 最終位置で埋める
            ],
            axis=0,
        )

        return {
            "observations": padded_obs,
            "actions": padded_actions,
            "rewards": padded_rewards,
            "dones": padded_dones,
            "positions": padded_positions,
        }


def main():
    parser = argparse.ArgumentParser(description="MiniGridデータ生成")
    parser.add_argument(
        "--env_name", type=str, default="MiniGrid-Empty-8x8-v0", help="MiniGrid環境名"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=1000, help="生成するエピソード数"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="出力ファイルパス (.npz)"
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="random",
        choices=["random"],
        help="ポリシータイプ",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="エピソードの最大ステップ数（指定しない場合は環境のデフォルト）",
    )
    parser.add_argument("--seed", type=int, default=42, help="ランダムシード")
    parser.add_argument(
        "--pad_length",
        type=int,
        default=None,
        help="エピソードをパディングする長さ（指定しない場合は可変長）",
    )
    parser.add_argument(
        "--render", action="store_true", help="レンダリングを有効化（デバッグ用）"
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="観測画像のリサイズサイズ（指定しない場合は元のサイズ、例: 64, 84）",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=8,
        help="MiniGridのタイルサイズ（デフォルト: 8）",
    )
    parser.add_argument(
        "--agent_view_size",
        type=int,
        default=None,
        help="エージェントの視界サイズ（指定しない場合は全体観測）",
    )

    args = parser.parse_args()

    # 出力ディレクトリを作成
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MiniGrid Data Generation")
    print("=" * 60)
    print(f"Environment: {args.env_name}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Output: {args.output_path}")
    print(f"Policy: {args.policy_type}")
    print(f"Max steps: {args.max_steps if args.max_steps else 'Default'}")
    print(f"Tile size: {args.tile_size}")
    print(
        f"Agent view size: {args.agent_view_size if args.agent_view_size else 'Full'}"
    )
    if args.resize is not None:
        print(f"Resize: {args.resize}x{args.resize}")
    else:
        print("Resize: None (original size)")
    print("=" * 60)

    # 環境の作成
    env_kwargs = {
        "render_mode": "human" if args.render else None,
        "tile_size": args.tile_size,
        "highlight": False,  # エージェント視野のハイライトを無効化
    }

    if args.max_steps is not None:
        env_kwargs["max_steps"] = args.max_steps

    if args.agent_view_size is not None:
        env_kwargs["agent_view_size"] = args.agent_view_size

    env = gym.make(args.env_name, **env_kwargs)

    # RGBImgObsWrapperで完全観測のRGB画像に変換
    env = RGBImgObsWrapper(env, tile_size=args.tile_size)
    # ImgObsWrapperで辞書から画像のみを取り出す
    env = ImgObsWrapper(env)

    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Max steps: {env.max_steps if hasattr(env, 'max_steps') else 'N/A'}")

    # ポリシーの作成
    if args.policy_type == "random":
        policy = create_random_policy(env)
    else:
        raise ValueError(f"Unknown policy type: {args.policy_type}")

    # データ生成
    all_episodes = []
    episode_lengths = []

    print(f"\nGenerating {args.n_episodes} episodes...")

    for ep_idx in tqdm(range(args.n_episodes), desc="Episodes"):
        episode = generate_episode(
            env, policy, max_steps=args.max_steps or 1000, seed=args.seed + ep_idx
        )

        episode_lengths.append(len(episode["observations"]))

        # リサイズ（必要な場合）
        if args.resize is not None:
            episode["observations"] = resize_observations(
                episode["observations"], args.resize
            )

        # パディング（必要な場合）
        if args.pad_length is not None:
            episode = pad_episode(episode, args.pad_length)

        all_episodes.append(episode)

    env.close()

    # 統計情報
    print("\n" + "=" * 60)
    print("Episode Statistics:")
    print("=" * 60)
    print(f"  Total episodes: {len(all_episodes)}")
    print(f"  Episode lengths:")
    print(f"    Mean: {np.mean(episode_lengths):.1f}")
    print(f"    Std: {np.std(episode_lengths):.1f}")
    print(f"    Min: {np.min(episode_lengths)}")
    print(f"    Max: {np.max(episode_lengths)}")

    # データを配列に変換
    if args.pad_length is not None:
        # パディング済み: 固定長配列
        observations = np.array([ep["observations"] for ep in all_episodes])
        actions = np.array([ep["actions"] for ep in all_episodes])
        rewards = np.array([ep["rewards"] for ep in all_episodes])
        dones = np.array([ep["dones"] for ep in all_episodes])
        positions = np.array([ep["positions"] for ep in all_episodes])
    else:
        # 可変長: object配列
        observations = np.array(
            [ep["observations"] for ep in all_episodes], dtype=object
        )
        actions = np.array([ep["actions"] for ep in all_episodes], dtype=object)
        rewards = np.array([ep["rewards"] for ep in all_episodes], dtype=object)
        dones = np.array([ep["dones"] for ep in all_episodes], dtype=object)
        positions = np.array([ep["positions"] for ep in all_episodes], dtype=object)

    # 保存
    print(f"\nSaving data to {args.output_path}...")
    np.savez_compressed(
        args.output_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        positions=positions,
    )

    # ファイルサイズ
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    # 形状情報
    print(f"\nSaved data shapes:")
    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Dones: {dones.shape}")
    print(f"  Positions: {positions.shape}")

    if args.pad_length is not None and len(all_episodes) > 0:
        print(f"\nSample episode shapes (after padding):")
        print(f"  Observations: {all_episodes[0]['observations'].shape}")
        print(f"  Actions: {all_episodes[0]['actions'].shape}")

    print("\n" + "=" * 60)
    print("Data generation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
