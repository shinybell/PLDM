"""
Atariデータセットの動作テストスクリプト

使用方法:
    python pldm_envs/atari/test_dataset.py
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pldm_envs.atari.enums import AtariDatasetConfig
from pldm_envs.atari.data.atari_dataset import AtariDataset
import matplotlib.pyplot as plt
import numpy as np


def test_dataset_with_dummy_data():
    """
    ダミーデータでデータセットをテスト
    """
    print("=" * 60)
    print("Atari Dataset Test (with dummy data)")
    print("=" * 60)

    # ダミーデータを作成
    n_episodes = 10
    episode_length = 50
    height, width = 210, 160  # Atari標準サイズ

    # ダミーデータ生成
    observations = np.random.randint(
        0, 255,
        size=(n_episodes, episode_length, height, width, 3),
        dtype=np.uint8
    )
    actions = np.random.randint(0, 9, size=(n_episodes, episode_length - 1))
    rewards = np.random.randn(n_episodes, episode_length - 1).astype(np.float32)
    dones = np.zeros((n_episodes, episode_length - 1), dtype=bool)
    dones[:, -1] = True  # 最後のステップで終了

    # ダミーデータを保存
    dummy_path = project_root / "test_atari_data.npz"
    np.savez(
        dummy_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )

    print(f"Created dummy data at: {dummy_path}")
    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")

    # データセット設定
    config = AtariDatasetConfig(
        env_name="ALE/MsPacman-v5",
        data_path=str(dummy_path),
        batch_size=4,
        sample_length=17,
        img_size=64,
        grayscale=False,
        frame_stack=1,
        normalize_images=True,
        include_rewards=True,
        include_dones=True,
        quick_debug=True,
    )

    # データセットを作成
    print("\n" + "-" * 60)
    print("Creating dataset...")
    print("-" * 60)

    try:
        dataset = AtariDataset(config)

        print(f"\nDataset created successfully!")
        print(f"  Dataset length: {len(dataset)}")

        # サンプルを取得
        print("\n" + "-" * 60)
        print("Testing sample retrieval...")
        print("-" * 60)

        sample = dataset[0]

        print(f"\nSample 0:")
        print(f"  States shape: {sample.states.shape}")
        print(f"  States dtype: {sample.states.dtype}")
        print(f"  States min/max: {sample.states.min():.3f} / {sample.states.max():.3f}")
        print(f"  Actions shape: {sample.actions.shape}")
        print(f"  Actions dtype: {sample.actions.dtype}")

        if sample.rewards is not None:
            print(f"  Rewards shape: {sample.rewards.shape}")
            print(f"  Rewards dtype: {sample.rewards.dtype}")

        if sample.dones is not None:
            print(f"  Dones shape: {sample.dones.shape}")
            print(f"  Dones dtype: {sample.dones.dtype}")

        # 可視化
        print("\n" + "-" * 60)
        print("Visualizing first 5 frames...")
        print("-" * 60)

        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, ax in enumerate(axes):
            if i < sample.states.shape[0]:
                # [C, H, W] -> [H, W, C]
                img = sample.states[i].permute(1, 2, 0).numpy()

                # グレースケールの場合
                if img.shape[2] == 1:
                    img = img[:, :, 0]
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)

                ax.set_title(f"t={i}")
                ax.axis('off')

        output_path = project_root / "test_atari_visualization.png"
        plt.savefig(output_path)
        print(f"\nSaved visualization to: {output_path}")

        # 複数サンプルをテスト
        print("\n" + "-" * 60)
        print("Testing multiple samples...")
        print("-" * 60)

        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"Sample {i}: states {sample.states.shape}, actions {sample.actions.shape}")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # クリーンアップ
        if dummy_path.exists():
            dummy_path.unlink()
            print(f"\nCleaned up dummy data: {dummy_path}")


def test_dataset_with_real_data(data_path: str):
    """
    実際のデータでデータセットをテスト

    Args:
        data_path: データファイルのパス
    """
    print("=" * 60)
    print("Atari Dataset Test (with real data)")
    print("=" * 60)

    config = AtariDatasetConfig(
        env_name="ALE/MsPacman-v5",
        data_path=data_path,
        batch_size=32,
        sample_length=17,
        img_size=64,
        quick_debug=False,
    )

    dataset = AtariDataset(config)

    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"  States: {sample.states.shape}")
    print(f"  Actions: {sample.actions.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to real Atari data file (.npz)"
    )
    args = parser.parse_args()

    if args.data_path:
        test_dataset_with_real_data(args.data_path)
    else:
        test_dataset_with_dummy_data()
