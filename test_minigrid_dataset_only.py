"""
Test MiniGrid Dataset directly (without DatasetFactory)

This is a simpler test that only tests the MiniGridDataset class itself.
"""

import sys
from pathlib import Path
import torch

# データパスの確認
data_path = Path("data/minigrid/test_level1_72x72.npz")
if not data_path.exists():
    print(f"❌ Error: Data file not found at {data_path}")
    sys.exit(1)

print(f"✓ Found data file: {data_path}")

# MiniGridデータセットのインポート
from pldm_envs.minigrid.data.minigrid_dataset import (
    MiniGridDataset,
    MiniGridDatasetConfig,
)
from pldm_envs.minigrid.data import minigrid_collate_fn

print("\n=== Testing MiniGrid Dataset ===\n")

# 設定
config = MiniGridDatasetConfig(
    data_path=str(data_path),
    sample_length=16,
    img_size=72,
    normalize_images=True,
    batch_size=4,
    train=True,
    quick_debug=True,
)

print(f"Data path: {config.data_path}")
print(f"Sample length: {config.sample_length}")
print(f"Image size: {config.img_size}")
print(f"Batch size: {config.batch_size}")

# データセット作成
try:
    print("\n--- Creating dataset ---")
    dataset = MiniGridDataset(config)

    print(f"✓ Dataset created successfully!")
    print(f"  Dataset length: {len(dataset)}")

    # サンプルを1つ取得
    print("\n--- Testing single sample ---")
    sample = dataset[0]
    print(f"  States shape: {sample.states.shape}")
    print(f"  Actions shape: {sample.actions.shape}")
    print(f"  States dtype: {sample.states.dtype}")
    print(f"  Actions dtype: {sample.actions.dtype}")
    print(f"  States range: [{sample.states.min():.3f}, {sample.states.max():.3f}]")

    # DataLoaderで複数サンプルをバッチ化
    print("\n--- Testing DataLoader with collate_fn ---")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=minigrid_collate_fn,
    )

    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  States shape: {batch.states.shape}")
        print(f"  Actions shape: {batch.actions.shape}")
        print(f"  States dtype: {batch.states.dtype}")
        print(f"  Actions dtype: {batch.actions.dtype}")
        print(f"  States range: [{batch.states.min():.3f}, {batch.states.max():.3f}]")

        # 2バッチだけテスト
        if batch_idx >= 1:
            break

    print("\n✓ All tests passed!")
    print("\n=== MiniGrid Dataset Test Successful ===")

except Exception as e:
    print(f"\n❌ Error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
