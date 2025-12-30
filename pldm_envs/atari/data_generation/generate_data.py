"""
Atari環境のデータ生成スクリプト

Gymnasiumを使ってAtari環境からデータを収集します。

使用方法:
    # ランダムポリシーでデータ生成
    python pldm_envs/atari/data_generation/generate_data.py \\
        --env_name MsPacman-v5 \\
        --n_episodes 1000 \\
        --output_path data/mspacman_random.npz

    # デバッグ用（少量データ）
    python pldm_envs/atari/data_generation/generate_data.py \\
        --env_name MsPacman-v5 \\
        --n_episodes 10 \\
        --output_path data/mspacman_debug.npz

インストール要件:
    pip install 'gymnasium[atari]'
"""

import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import gymnasium as gym


def generate_episode(env, policy, max_steps=10000, seed=None):
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
    """
    obs_list = []
    action_list = []
    reward_list = []
    done_list = []

    obs, info = env.reset(seed=seed)
    obs_list.append(obs)

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        obs_list.append(obs)
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(terminated or truncated)

        if terminated or truncated:
            break

    return {
        'observations': np.array(obs_list),
        'actions': np.array(action_list),
        'rewards': np.array(reward_list),
        'dones': np.array(done_list),
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
        target_length: 目標長

    Returns:
        パディングされたエピソード
    """
    current_length = len(episode['observations'])

    if current_length >= target_length:
        # 切り詰め
        return {
            'observations': episode['observations'][:target_length],
            'actions': episode['actions'][:target_length-1],
            'rewards': episode['rewards'][:target_length-1],
            'dones': episode['dones'][:target_length-1],
        }
    else:
        # パディング
        pad_length = target_length - current_length

        # 最後のフレームを繰り返してパディング
        last_obs = episode['observations'][-1]
        padded_obs = np.concatenate([
            episode['observations'],
            np.repeat(last_obs[np.newaxis], pad_length, axis=0)
        ], axis=0)

        # アクション、報酬、doneは0でパディング
        padded_actions = np.concatenate([
            episode['actions'],
            np.zeros(pad_length, dtype=episode['actions'].dtype)
        ], axis=0)

        padded_rewards = np.concatenate([
            episode['rewards'],
            np.zeros(pad_length, dtype=episode['rewards'].dtype)
        ], axis=0)

        padded_dones = np.concatenate([
            episode['dones'],
            np.ones(pad_length, dtype=bool)  # パディング部分は終了扱い
        ], axis=0)

        return {
            'observations': padded_obs,
            'actions': padded_actions,
            'rewards': padded_rewards,
            'dones': padded_dones,
        }


def main():
    parser = argparse.ArgumentParser(description='Atariデータ生成')
    parser.add_argument('--env_name', type=str, default='ALE/Pacman-v5',
                        help='Atari環境名')
    parser.add_argument('--n_episodes', type=int, default=1000,
                        help='生成するエピソード数')
    parser.add_argument('--output_path', type=str, required=True,
                        help='出力ファイルパス (.npz)')
    parser.add_argument('--policy_type', type=str, default='random',
                        choices=['random'],
                        help='ポリシータイプ')
    parser.add_argument('--max_episode_steps', type=int, default=10000,
                        help='エピソードの最大ステップ数')
    parser.add_argument('--seed', type=int, default=42,
                        help='ランダムシード')
    parser.add_argument('--obs_type', type=str, default='rgb',
                        choices=['rgb', 'grayscale'],
                        help='観測タイプ')
    parser.add_argument('--frameskip', type=int, default=4,
                        help='フレームスキップ')
    parser.add_argument('--pad_length', type=int, default=None,
                        help='エピソードをパディングする長さ（指定しない場合は可変長）')
    parser.add_argument('--render', action='store_true',
                        help='レンダリングを有効化（デバッグ用）')

    args = parser.parse_args()

    # 出力ディレクトリを作成
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Atari Data Generation")
    print("=" * 60)
    print(f"Environment: {args.env_name}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Output: {args.output_path}")
    print(f"Policy: {args.policy_type}")
    print(f"Obs type: {args.obs_type}")
    print(f"Frameskip: {args.frameskip}")
    print("=" * 60)

    # 環境の作成
    env = gym.make(
        args.env_name,
        obs_type=args.obs_type,
        frameskip=args.frameskip,
        render_mode='human' if args.render else None,
    )

    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # ポリシーの作成
    if args.policy_type == 'random':
        policy = create_random_policy(env)
    else:
        raise ValueError(f"Unknown policy type: {args.policy_type}")

    # データ生成
    all_episodes = []
    episode_lengths = []

    print(f"\nGenerating {args.n_episodes} episodes...")

    for ep_idx in tqdm(range(args.n_episodes), desc="Episodes"):
        episode = generate_episode(
            env,
            policy,
            max_steps=args.max_episode_steps,
            seed=args.seed + ep_idx
        )

        episode_lengths.append(len(episode['observations']))

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
        observations = np.array([ep['observations'] for ep in all_episodes])
        actions = np.array([ep['actions'] for ep in all_episodes])
        rewards = np.array([ep['rewards'] for ep in all_episodes])
        dones = np.array([ep['dones'] for ep in all_episodes])
    else:
        # 可変長: object配列
        observations = np.array([ep['observations'] for ep in all_episodes], dtype=object)
        actions = np.array([ep['actions'] for ep in all_episodes], dtype=object)
        rewards = np.array([ep['rewards'] for ep in all_episodes], dtype=object)
        dones = np.array([ep['dones'] for ep in all_episodes], dtype=object)

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
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Dones: {dones.shape}")

    if args.pad_length is not None and len(all_episodes) > 0:
        print(f"\nSample episode shapes (after padding):")
        print(f"  Observations: {all_episodes[0]['observations'].shape}")
        print(f"  Actions: {all_episodes[0]['actions'].shape}")

    print("\n" + "=" * 60)
    print("Data generation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
