#!/usr/bin/env python3
"""
MiniGrid LongHorizon環境用のPLDMデータ生成スクリプト

PLDMWrapperを使用して、64x64または72x72のRGB観測データを生成します。

使用例:
    # Level 1で1000エピソード生成（64x64）
    python pldm_envs/minigrid/data_generation/generate_pldm_data.py \\
        --env_name MiniGrid-LongHorizon-Level1-v0 \\
        --n_episodes 1000 \\
        --output_path data/minigrid/level1_train.npz

    # Level 2で1000エピソード生成（72x72）
    python pldm_envs/minigrid/data_generation/generate_pldm_data.py \\
        --env_name MiniGrid-LongHorizon-Level2-v0 \\
        --n_episodes 1000 \\
        --obs_size 72 \\
        --output_path data/minigrid/level2_train_72x72.npz

    # PyTorch形式（CHW, normalized）
    python pldm_envs/minigrid/data_generation/generate_pldm_data.py \\
        --env_name MiniGrid-LongHorizon-Level1-v0 \\
        --n_episodes 1000 \\
        --channel_first \\
        --normalize \\
        --output_path data/minigrid/level1_train_pytorch.npz
"""

import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import gymnasium as gym

# Register MiniGrid environments
import pldm_envs.minigrid
from pldm_envs.minigrid.wrappers import make_pldm_env


def generate_episode(env, policy, max_steps=None, seed=None):
    """
    1エピソードのデータを生成

    Args:
        env: PLDM Wrapper適用済みのGymnasium環境
        policy: ポリシー関数 (obs -> action)
        max_steps: 最大ステップ数（Noneの場合は環境のデフォルト）
        seed: ランダムシード

    Returns:
        dict: エピソードデータ
            - observations: [T, H, W, C] or [T, C, H, W]
            - actions: [T-1]
            - rewards: [T-1]
            - dones: [T-1]
    """
    obs_list = []
    action_list = []
    reward_list = []
    done_list = []

    obs, info = env.reset(seed=seed)
    obs_list.append(obs)

    # 環境のmax_stepsを取得
    env_max_steps = max_steps or getattr(env.unwrapped, 'max_steps', 256)

    for step in range(env_max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(terminated or truncated)

        if terminated or truncated:
            break

    return {
        "observations": np.array(obs_list),
        "actions": np.array(action_list),
        "rewards": np.array(reward_list),
        "dones": np.array(done_list),
    }


def create_random_policy(env):
    """ランダムポリシーを作成"""
    def policy(obs):
        return env.action_space.sample()
    return policy


def pad_episode(episode, target_length):
    """
    エピソードを指定長にパディング

    Args:
        episode: エピソードデータ
        target_length: 目標長（観測数）

    Returns:
        パディングされたエピソード
    """
    current_length = len(episode["observations"])

    if current_length >= target_length:
        # 切り詰め
        return {
            "observations": episode["observations"][:target_length],
            "actions": episode["actions"][:target_length - 1],
            "rewards": episode["rewards"][:target_length - 1],
            "dones": episode["dones"][:target_length - 1],
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

        return {
            "observations": padded_obs,
            "actions": padded_actions,
            "rewards": padded_rewards,
            "dones": padded_dones,
        }


def main():
    parser = argparse.ArgumentParser(description="MiniGrid PLDM Data Generation")
    parser.add_argument(
        "--env_name",
        type=str,
        default="MiniGrid-LongHorizon-Level1-v0",
        help="MiniGrid環境名 (Level1/2/3)",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=1000, help="生成するエピソード数"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="出力ファイルパス (.npz)"
    )
    parser.add_argument(
        "--obs_size",
        type=int,
        default=72,
        help="観測画像のサイズ (64, 72, 84など)",
    )
    parser.add_argument(
        "--channel_first",
        action="store_true",
        help="チャンネルファーストフォーマット (C, H, W) を使用",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="観測を[0, 1]に正規化（float32）",
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

    args = parser.parse_args()

    # 出力ディレクトリを作成
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MiniGrid PLDM Data Generation")
    print("=" * 60)
    print(f"Environment: {args.env_name}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Output: {args.output_path}")
    print(f"Observation size: {args.obs_size}x{args.obs_size}")
    print(f"Channel first: {args.channel_first}")
    print(f"Normalize: {args.normalize}")
    print(f"Policy: {args.policy_type}")
    print(f"Max steps: {args.max_steps if args.max_steps else 'Default'}")
    if args.pad_length:
        print(f"Pad length: {args.pad_length}")
    print("=" * 60)

    # 環境の作成（PLDMWrapper適用）
    env = make_pldm_env(
        args.env_name,
        size=(args.obs_size, args.obs_size),
        channel_first=args.channel_first,
        normalize=args.normalize,
    )

    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    unwrapped = env.unwrapped
    print(f"  Grid size: {unwrapped.width}x{unwrapped.height}")
    print(f"  Max steps: {unwrapped.max_steps}")

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
            env, policy, max_steps=args.max_steps, seed=args.seed + ep_idx
        )

        episode_lengths.append(len(episode["observations"]))

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
    else:
        # 可変長: object配列
        observations = np.array(
            [ep["observations"] for ep in all_episodes], dtype=object
        )
        actions = np.array([ep["actions"] for ep in all_episodes], dtype=object)
        rewards = np.array([ep["rewards"] for ep in all_episodes], dtype=object)
        dones = np.array([ep["dones"] for ep in all_episodes], dtype=object)

    # 保存
    print(f"\nSaving data to {args.output_path}...")
    np.savez_compressed(
        args.output_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )

    # ファイルサイズ
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    # 形状情報
    print(f"\nSaved data shapes:")
    print(f"  Observations: {observations.shape}")
    if args.pad_length is not None and len(all_episodes) > 0:
        print(f"  First episode obs shape: {all_episodes[0]['observations'].shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Dones: {dones.shape}")

    # サンプルデータの値範囲
    if len(all_episodes) > 0:
        sample_obs = all_episodes[0]['observations'][0]
        print(f"\nSample observation:")
        print(f"  Shape: {sample_obs.shape}")
        print(f"  Dtype: {sample_obs.dtype}")
        print(f"  Value range: [{sample_obs.min():.3f}, {sample_obs.max():.3f}]")

    print("\n" + "=" * 60)
    print("Data generation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
