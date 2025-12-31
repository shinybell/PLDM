"""
Test MiniGrid integration with PLDM DatasetFactory

This script tests whether MiniGrid datasets can be created through
the PLDM DatasetFactory system.
"""

import sys
from pathlib import Path

# データパスの確認（既存のテストデータを使用）
data_path = Path("data/minigrid/test_level1_72x72.npz")
if not data_path.exists():
    print(f"❌ Error: Data file not found at {data_path}")
    print("Please generate data first using:")
    print("  python pldm_envs/minigrid/data_generation/generate_pldm_data.py \\")
    print("    --env_name MiniGrid-LongHorizon-Level1-v0 \\")
    print("    --n_episodes 100 \\")
    print("    --obs_size 72 \\")
    print("    --output_path data/minigrid/test_level1_72x72.npz")
    sys.exit(1)

print(f"✓ Found data file: {data_path}")

# PLDM設定のインポート
from pldm.data.enums import DataConfig, DatasetType
from pldm.data.dataset_factory import DatasetFactory
from pldm_envs.minigrid.enums import MiniGridDatasetConfig

print("\n=== Testing MiniGrid PLDM Integration ===\n")

# MiniGrid設定
minigrid_config = MiniGridDatasetConfig(
    data_path=str(data_path),
    sample_length=16,
    img_size=72,
    normalize_images=True,
    batch_size=4,
    train=True,
    quick_debug=True,  # 高速化のため
)

# DataConfig作成
data_config = DataConfig(
    dataset_type=DatasetType.MiniGrid,
    minigrid_config=minigrid_config,
    normalize=False,  # 画像は既に正規化されている
    num_workers=0,
    quick_debug=True,
)

print(f"Dataset type: {data_config.dataset_type}")
print(f"Data path: {data_config.minigrid_config.data_path}")
print(f"Sample length: {data_config.minigrid_config.sample_length}")
print(f"Image size: {data_config.minigrid_config.img_size}")
print(f"Batch size: {data_config.minigrid_config.batch_size}")

# DatasetFactory経由でデータセット作成
try:
    print("\n--- Creating datasets via DatasetFactory ---")
    factory = DatasetFactory(config=data_config)
    datasets = factory.create_datasets()

    print(f"✓ Successfully created datasets!")
    print(f"  Train dataset: {type(datasets.ds)}")
    print(f"  Val dataset: {datasets.val_ds}")

    # データローダーからサンプルを取得
    print("\n--- Testing data loading ---")
    batch_count = 0
    for batch in datasets.ds:
        print(f"\nBatch {batch_count + 1}:")
        print(f"  States shape: {batch.states.shape}")
        print(f"  Actions shape: {batch.actions.shape}")
        print(f"  States dtype: {batch.states.dtype}")
        print(f"  Actions dtype: {batch.actions.dtype}")
        print(f"  States range: [{batch.states.min():.3f}, {batch.states.max():.3f}]")

        # 1バッチだけテスト
        batch_count += 1
        if batch_count >= 2:
            break

    print("\n✓ All tests passed!")
    print("\n=== MiniGrid PLDM Integration Successful ===")

except Exception as e:
    print(f"\n❌ Error during dataset creation:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
