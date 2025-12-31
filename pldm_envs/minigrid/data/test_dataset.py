#!/usr/bin/env python3
"""
MiniGridDatasetのテストスクリプト

生成したオフラインデータが正しくロードできるかテストします。

使用方法:
    python pldm_envs/minigrid/data/test_dataset.py \
        --data_path data/minigrid/test_level1_72x72.npz
"""

import argparse
import torch
from torch.utils.data import DataLoader

from pldm_envs.minigrid.data import (
    MiniGridDataset,
    MiniGridDatasetConfig,
    MiniGridSample,
)


def test_dataset(data_path: str, sample_length: int = 16, img_size: int = 72):
    """データセットをテスト"""
    print("=" * 60)
    print("MiniGrid Dataset Test")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Sample length: {sample_length}")
    print(f"Image size: {img_size}x{img_size}")
    print("=" * 60)

    # データセット作成
    config = MiniGridDatasetConfig(
        data_path=data_path,
        sample_length=sample_length,
        img_size=img_size,
        normalize_images=True,
    )

    dataset = MiniGridDataset(config)

    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Total samples: {len(dataset)}")

    # サンプルを取得
    print("\nTesting sample retrieval...")
    sample = dataset[0]

    print(f"  states: {sample.states.shape} {sample.states.dtype}")
    print(f"  actions: {sample.actions.shape} {sample.actions.dtype}")
    print(f"  Value range: [{sample.states.min():.3f}, {sample.states.max():.3f}]")

    # 期待される形状を確認
    expected_states_shape = (sample_length, 3, img_size, img_size)
    expected_actions_shape = (sample_length - 1, 1)

    assert sample.states.shape == expected_states_shape, \
        f"States shape mismatch: {sample.states.shape} != {expected_states_shape}"
    assert sample.actions.shape == expected_actions_shape, \
        f"Actions shape mismatch: {sample.actions.shape} != {expected_actions_shape}"
    assert sample.states.dtype == torch.float32, \
        f"States dtype mismatch: {sample.states.dtype}"
    assert sample.actions.dtype == torch.int64, \
        f"Actions dtype mismatch: {sample.actions.dtype}"

    print("✓ Sample shapes and dtypes are correct!")

    # カスタムcollate関数
    def collate_fn(batch):
        """MiniGridSampleのバッチをまとめる"""
        states = torch.stack([sample.states for sample in batch])
        actions = torch.stack([sample.actions for sample in batch])

        # rewardsとdonesは全てNoneまたは全て有効値
        if batch[0].rewards is not None:
            rewards = torch.stack([sample.rewards for sample in batch])
        else:
            rewards = None

        if batch[0].dones is not None:
            dones = torch.stack([sample.dones for sample in batch])
        else:
            dones = None

        return MiniGridSample(
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

    # DataLoaderをテスト
    print("\nTesting DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    batch = next(iter(dataloader))
    print(f"  Batch states: {batch.states.shape} {batch.states.dtype}")
    print(f"  Batch actions: {batch.actions.shape} {batch.actions.dtype}")

    expected_batch_states = (4, sample_length, 3, img_size, img_size)
    expected_batch_actions = (4, sample_length - 1, 1)

    assert batch.states.shape == expected_batch_states, \
        f"Batch states shape mismatch: {batch.states.shape} != {expected_batch_states}"
    assert batch.actions.shape == expected_batch_actions, \
        f"Batch actions shape mismatch: {batch.actions.shape} != {expected_batch_actions}"

    print("✓ DataLoader works correctly!")

    # 複数バッチをテスト
    print("\nIterating through batches...")
    num_batches = min(5, len(dataloader))
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        print(f"  Batch {i+1}: states={batch.states.shape}, actions={batch.actions.shape}")

    print(f"✓ Successfully iterated through {num_batches} batches!")

    # 統計情報
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    # 最初の100サンプルから統計を取る
    num_samples_for_stats = min(100, len(dataset))
    all_states = []
    all_actions = []

    for i in range(num_samples_for_stats):
        sample = dataset[i]
        all_states.append(sample.states)
        all_actions.append(sample.actions)

    all_states = torch.stack(all_states)
    all_actions = torch.cat(all_actions)

    print(f"Statistics from first {num_samples_for_stats} samples:")
    print(f"  States:")
    print(f"    Mean: {all_states.mean():.3f}")
    print(f"    Std: {all_states.std():.3f}")
    print(f"    Min: {all_states.min():.3f}")
    print(f"    Max: {all_states.max():.3f}")
    print(f"  Actions:")
    print(f"    Unique values: {torch.unique(all_actions).tolist()}")
    print(f"    Distribution: {torch.bincount(all_actions.squeeze()).tolist()}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test MiniGrid Dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/minigrid/test_level1_72x72.npz",
        help="Path to .npz data file",
    )
    parser.add_argument(
        "--sample_length",
        type=int,
        default=16,
        help="Sample length (context length)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=72,
        help="Image size (64, 72, etc.)",
    )

    args = parser.parse_args()

    test_dataset(
        data_path=args.data_path,
        sample_length=args.sample_length,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
