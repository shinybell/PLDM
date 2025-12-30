"""
MiniGrid環境とデータセットの動作確認スクリプト

使用方法:
    # 環境のテスト
    python pldm_envs/minigrid/test_minigrid.py --test_env

    # データ生成のテスト
    python pldm_envs/minigrid/test_minigrid.py --test_generation

    # データセットのテスト
    python pldm_envs/minigrid/test_minigrid.py --test_dataset --data_path data/minigrid/empty_8x8_debug.npz
"""

import argparse
import sys
from pathlib import Path


def test_environment():
    """MiniGrid環境のテスト"""
    print("=" * 60)
    print("Testing MiniGrid Environment")
    print("=" * 60)

    try:
        import gymnasium as gym
        import numpy as np
        import minigrid
        from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

        gym.register_envs(minigrid)
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install: pip install minigrid")
        return False

    # 環境の作成
    env = gym.make(
        "MiniGrid-Empty-8x8-v0",
        max_steps=200,
        tile_size=8,
        render_mode=None,
    )
    # RGBImgObsWrapperで完全観測のRGB画像に変換
    env = RGBImgObsWrapper(env)
    # ImgObsWrapperで辞書から画像のみを取り出す
    env = ImgObsWrapper(env)

    print(f"✓ Environment created successfully")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Max steps: {env.max_steps if hasattr(env, 'max_steps') else 'N/A'}")

    # エピソードの実行
    obs, info = env.reset(seed=42)
    print(f"  Initial observation shape: {obs.shape}")

    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}")
            break

    print(f"  Total reward: {total_reward}")
    print(f"✓ Environment test passed!\n")

    env.close()
    return True


def test_data_generation():
    """データ生成のテスト"""
    print("=" * 60)
    print("Testing Data Generation")
    print("=" * 60)

    import subprocess
    import tempfile

    # 一時ファイルを作成
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        output_path = tmp.name

    try:
        # データ生成スクリプトを実行
        cmd = [
            sys.executable,
            "pldm_envs/minigrid/data_generation/generate_data.py",
            "--env_name",
            "MiniGrid-Empty-8x8-v0",
            "--n_episodes",
            "5",
            "--max_steps",
            "200",
            "--resize",
            "64",
            "--output_path",
            output_path,
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: Data generation failed")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

        print(f"✓ Data generation completed")

        # データの確認
        import numpy as np

        data = np.load(output_path, allow_pickle=True)

        print(f"  Observations shape: {data['observations'].shape}")
        print(f"  Actions shape: {data['actions'].shape}")
        print(f"✓ Data generation test passed!\n")

        return True

    finally:
        # 一時ファイルを削除
        Path(output_path).unlink(missing_ok=True)


def test_dataset(data_path):
    """データセットのテスト"""
    print("=" * 60)
    print("Testing MiniGrid Dataset")
    print("=" * 60)

    try:
        import torch
        from pldm_envs.minigrid.enums import MiniGridDatasetConfig
        from pldm_envs.minigrid.data.minigrid_dataset import MiniGridDataset
    except ImportError as e:
        print(f"Error: {e}")
        return False

    if not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please run with --test_generation first to generate test data")
        return False

    # データセット設定
    config = MiniGridDatasetConfig(
        env_name="MiniGrid-Empty-8x8-v0",
        data_path=data_path,
        batch_size=4,
        sample_length=17,
        img_size=64,
        normalize_images=True,
        train=True,
    )

    # データセットの作成
    dataset = MiniGridDataset(config)

    print(f"✓ Dataset created successfully")
    print(f"  Dataset length: {len(dataset)}")

    # サンプルの取得
    sample = dataset[0]

    print(f"  Sample states shape: {sample.states.shape}")
    print(f"  Sample actions shape: {sample.actions.shape}")
    print(f"  States dtype: {sample.states.dtype}")
    print(f"  Actions dtype: {sample.actions.dtype}")
    print(f"  States range: [{sample.states.min():.3f}, {sample.states.max():.3f}]")

    # DataLoaderのテスト（カスタムcollate関数）
    def collate_fn(batch):
        """カスタムcollate関数（Noneを含むNamedTupleに対応）"""
        states = torch.stack([sample.states for sample in batch])
        actions = torch.stack([sample.actions for sample in batch])

        # rewards/donesがある場合はスタック、ない場合はNone
        rewards = None
        if batch[0].rewards is not None:
            rewards = torch.stack([sample.rewards for sample in batch])

        dones = None
        if batch[0].dones is not None:
            dones = torch.stack([sample.dones for sample in batch])

        return type(batch[0])(states, actions, rewards, dones)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    batch = next(iter(dataloader))
    print(f"  Batch states shape: {batch.states.shape}")
    print(f"  Batch actions shape: {batch.actions.shape}")
    print(f"✓ Dataset test passed!\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="MiniGrid環境のテスト")
    parser.add_argument("--test_env", action="store_true", help="環境のテスト")
    parser.add_argument(
        "--test_generation", action="store_true", help="データ生成のテスト"
    )
    parser.add_argument(
        "--test_dataset", action="store_true", help="データセットのテスト"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/minigrid/empty_8x8_debug.npz",
        help="データセットのパス",
    )
    parser.add_argument("--all", action="store_true", help="全テストを実行")

    args = parser.parse_args()

    # 引数が何も指定されていない場合は全テスト実行
    if not any([args.test_env, args.test_generation, args.test_dataset, args.all]):
        args.all = True

    success = True

    if args.all or args.test_env:
        success &= test_environment()

    if args.all or args.test_generation:
        success &= test_data_generation()

    if args.all or args.test_dataset:
        success &= test_dataset(args.data_path)

    if success:
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("✗ Some tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
