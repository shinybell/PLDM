"""
YAMLファイルからMiniGridのデータを生成するスクリプト

使用方法:
    python pldm_envs/minigrid/data_generation/generate_data_from_config.py \\
        --config pldm_envs/minigrid/configs/sample_10episodes.yaml \\
        --output_path pldm_envs/minigrid/data/sample_10episodes
"""

import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
import yaml
import torch

gym.register_envs(minigrid)

# カスタム環境を登録
import pldm_envs.minigrid

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    import warnings
    warnings.warn("opencv-python not found. Resizing will use a slower fallback method.")


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
            - observations: 不要（画像は後で生成）
            - actions: [T-1]
            - rewards: [T-1]
            - dones: [T-1]
            - positions: [T, 2] - エージェント位置 (x, y)
            - directions: [T] - エージェント向き (0-3)
    """
    action_list = []
    reward_list = []
    done_list = []
    position_list = []
    direction_list = []

    obs, info = env.reset(seed=seed)

    # 初期状態を記録
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    position_list.append(np.array(agent_pos, dtype=np.float32))
    direction_list.append(agent_dir)

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        action_list.append(action)
        reward_list.append(reward)
        done_list.append(terminated or truncated)

        # エージェント位置と向きを記録
        agent_pos = env.unwrapped.agent_pos
        agent_dir = env.unwrapped.agent_dir
        position_list.append(np.array(agent_pos, dtype=np.float32))
        direction_list.append(agent_dir)

        if terminated or truncated:
            break

    return {
        "actions": np.array(action_list),
        "rewards": np.array(reward_list),
        "dones": np.array(done_list),
        "positions": np.array(position_list, dtype=np.float32),
        "directions": np.array(direction_list, dtype=np.int32),
    }


def create_random_policy(env):
    """ランダムポリシーを作成"""
    def policy(obs):
        return env.action_space.sample()
    return policy


def main():
    parser = argparse.ArgumentParser(description="YAML設定からMiniGridデータを生成")
    parser.add_argument("--config", type=str, required=True, help="YAML設定ファイルパス")
    parser.add_argument("--output_path", type=str, required=True, help="出力ディレクトリパス")

    args = parser.parse_args()

    # 設定を読み込み
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # パラメータを取得
    env_name = config["env_name"]
    n_episodes = config["n_episodes"]
    episode_length = config["episode_length"]
    tile_size = config.get("tile_size", 8)
    agent_view_size = config.get("agent_view_size", None)
    seed = config.get("seed", 42)
    render = config.get("render", False)
    policy_type = config.get("policy_type", "random")

    # 出力ディレクトリを作成
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MiniGrid Data Generation from Config")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {n_episodes}")
    print(f"Episode length: {episode_length}")
    print(f"Tile size: {tile_size}")
    print(f"Agent view size: {agent_view_size if agent_view_size else 'Full'}")
    print(f"Policy: {policy_type}")
    print(f"Seed: {seed}")
    print(f"Output: {args.output_path}")
    print("=" * 60)

    # 環境を作成
    env_kwargs = {
        "render_mode": "human" if render else None,
        "tile_size": tile_size,
        "highlight": False,
        "max_steps": episode_length,
    }

    if agent_view_size is not None:
        env_kwargs["agent_view_size"] = agent_view_size

    env = gym.make(env_name, **env_kwargs)

    # 画像観測用のラッパーは不要（後でレンダリングするため）
    # ここではプロプリオセプティブデータのみ収集

    print(f"\nEnvironment created:")
    print(f"  Action space: {env.action_space}")
    print(f"  Max steps: {episode_length}")

    # ポリシーを作成
    if policy_type == "random":
        policy = create_random_policy(env)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    # データ生成
    all_episodes = []
    episode_lengths = []

    print(f"\nGenerating {n_episodes} episodes...")

    for ep_idx in tqdm(range(n_episodes), desc="Episodes"):
        episode = generate_episode(
            env, policy, max_steps=episode_length, seed=seed + ep_idx
        )

        episode_lengths.append(len(episode["positions"]))
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

    # データを保存（PyTorchのdata.p形式）
    data_path = output_path / "data.p"
    print(f"\nSaving data to {data_path}...")
    torch.save(all_episodes, data_path)

    # メタデータを保存
    metadata_path = output_path / "metadata.pt"
    print(f"Saving metadata to {metadata_path}...")
    torch.save(config, metadata_path)

    # ファイルサイズ
    file_size_mb = data_path.stat().st_size / (1024 * 1024)
    print(f"  Data file size: {file_size_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("Data generation completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Render images: python pldm_envs/minigrid/data_generation/render_data.py --data_path", args.output_path)
    print("  2. Postprocess images: python pldm_envs/minigrid/data_generation/postprocess_images.py --data_path", args.output_path)


if __name__ == "__main__":
    main()
