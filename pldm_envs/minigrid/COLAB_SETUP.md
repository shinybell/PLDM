# Google Colab で MiniGrid PLDM 学習を実行する方法

Google Colab環境でMiniGridのPLDM学習を実行するための完全ガイドです。

## 問題: Google DriveとMemory Mapの非互換性

Google Driveのファイルシステムは `mmap` (memory-mapped files) をサポートしていないため、Google Drive上のデータを直接読み込むとエラーが発生します。

**解決策**: データをColab のローカルストレージ (`/content/`) にコピーしてから学習を実行します。

---

## セットアップ手順

### 1. Google Driveをマウント

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. プロジェクトディレクトリに移動

```python
import os
os.chdir('/content/drive/MyDrive/weblab/world_model_2025/final_research/PLDM')
```

### 3. データをローカルにコピー

```python
import shutil
from pathlib import Path

# ソースパス（Google Drive上のデータ）
source_train_dir = Path("/content/drive/MyDrive/weblab/world_model_2025/final_research/minigrid_data/level1/train")
source_val_file = Path("/content/drive/MyDrive/weblab/world_model_2025/final_research/minigrid_data/level1/val.npz")

# コピー先パス（Colabローカルストレージ）
local_data_dir = Path("/content/minigrid_data/level1")

# ディレクトリ作成
local_data_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Copying MiniGrid data to local storage...")
print("=" * 60)

# 訓練データをコピー
if source_train_dir.is_dir():
    print(f"\nCopying training data from:")
    print(f"  {source_train_dir}")
    print(f"To:")
    print(f"  {local_data_dir / 'train'}")

    # ディレクトリをコピー（時間がかかる場合があります）
    if (local_data_dir / 'train').exists():
        print("  Training data already exists, skipping...")
    else:
        shutil.copytree(source_train_dir, local_data_dir / 'train')
        print("  ✓ Training data copied successfully")
else:
    print(f"ERROR: Training data not found at {source_train_dir}")

# 検証データをコピー
if source_val_file.exists():
    print(f"\nCopying validation data from:")
    print(f"  {source_val_file}")
    print(f"To:")
    print(f"  {local_data_dir / 'val.npz'}")

    if (local_data_dir / 'val.npz').exists():
        print("  Validation data already exists, skipping...")
    else:
        shutil.copy(source_val_file, local_data_dir / 'val.npz')
        print("  ✓ Validation data copied successfully")
else:
    print(f"ERROR: Validation data not found at {source_val_file}")

print("\n" + "=" * 60)
print("Data copy completed!")
print("=" * 60)

# データの確認
print("\nVerifying copied data...")
print(f"Training data exists: {(local_data_dir / 'train').exists()}")
print(f"Validation data exists: {(local_data_dir / 'val.npz').exists()}")

# 訓練データの詳細を確認
if (local_data_dir / 'train').is_dir():
    train_files = list((local_data_dir / 'train').glob('*.npy'))
    print(f"\nTraining data files ({len(train_files)}):")
    for f in sorted(train_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.1f} MB")
```

### 4. データの内容を確認（オプション）

```python
import numpy as np

# 訓練データの確認
train_obs = np.load('/content/minigrid_data/level1/train/observations.npy', mmap_mode='r')
train_actions = np.load('/content/minigrid_data/level1/train/actions.npy', mmap_mode='r')
train_positions = np.load('/content/minigrid_data/level1/train/positions.npy', mmap_mode='r')

print("=" * 60)
print("Training Data Summary")
print("=" * 60)
print(f"Observations shape: {train_obs.shape}")
print(f"  - Episodes: {train_obs.shape[0]}")
print(f"  - Timesteps: {train_obs.shape[1]}")
print(f"  - Image size: {train_obs.shape[2]}x{train_obs.shape[3]}")
print(f"  - Channels: {train_obs.shape[4]}")
print(f"  - Dtype: {train_obs.dtype}")
print(f"  - Value range: [{train_obs.min()}, {train_obs.max()}]")

print(f"\nActions shape: {train_actions.shape}")
print(f"  - Dtype: {train_actions.dtype}")

print(f"\nPositions shape: {train_positions.shape}")
print(f"  - Dtype: {train_positions.dtype}")

# 検証データの確認
val_data = np.load('/content/minigrid_data/level1/val.npz', allow_pickle=True)
print("\n" + "=" * 60)
print("Validation Data Summary")
print("=" * 60)
print(f"Keys: {list(val_data.keys())}")
print(f"Episodes: {len(val_data['observations'])}")
```

---

## 学習の実行

### オプション1: %%bash マジックを使用（推奨）

```bash
%%bash
PYTHONPATH=$PWD:$PYTHONPATH python pldm/train.py \
  --configs pldm/configs/minigrid/level1_test.yaml \
  --values \
    data.minigrid_config.data_path=/content/minigrid_data/level1/train \
    data.minigrid_config.val_path=/content/minigrid_data/level1/val.npz \
    output_root=/content/drive/MyDrive/weblab/world_model_2025/final_research/output/ \
    wandb=true
```

### オプション2: Pythonから実行

```python
import os
import subprocess

# 環境変数設定
os.environ['PYTHONPATH'] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"

# コマンド実行
cmd = [
    "python", "pldm/train.py",
    "--configs", "pldm/configs/minigrid/level1_test.yaml",
    "--values",
    "data.minigrid_config.data_path=/content/minigrid_data/level1/train",
    "data.minigrid_config.val_path=/content/minigrid_data/level1/val.npz",
    "output_root=/content/drive/MyDrive/weblab/world_model_2025/final_research/output/",
    "wandb=true"
]

result = subprocess.run(cmd, capture_output=False, text=True)
```

---

## 完全なワークフロー（コピー&ペースト可能）

```python
# ========================================
# 1. セットアップ
# ========================================
from google.colab import drive
import os
import shutil
from pathlib import Path
import numpy as np

# Google Driveマウント
drive.mount('/content/drive')

# プロジェクトディレクトリに移動
os.chdir('/content/drive/MyDrive/weblab/world_model_2025/final_research/PLDM')

# ========================================
# 2. データをローカルにコピー
# ========================================
source_train_dir = Path("/content/drive/MyDrive/weblab/world_model_2025/final_research/minigrid_data/level1/train")
source_val_file = Path("/content/drive/MyDrive/weblab/world_model_2025/final_research/minigrid_data/level1/val.npz")
local_data_dir = Path("/content/minigrid_data/level1")
local_data_dir.mkdir(parents=True, exist_ok=True)

print("Copying training data...")
if not (local_data_dir / 'train').exists():
    shutil.copytree(source_train_dir, local_data_dir / 'train')
    print("✓ Training data copied")
else:
    print("✓ Training data already exists")

print("Copying validation data...")
if not (local_data_dir / 'val.npz').exists():
    shutil.copy(source_val_file, local_data_dir / 'val.npz')
    print("✓ Validation data copied")
else:
    print("✓ Validation data already exists")

# ========================================
# 3. データ確認
# ========================================
print("\n" + "=" * 60)
print("Data Verification")
print("=" * 60)

train_obs = np.load('/content/minigrid_data/level1/train/observations.npy', mmap_mode='r')
print(f"Training observations: {train_obs.shape}")
print(f"Episodes: {train_obs.shape[0]}, Timesteps: {train_obs.shape[1]}")
print(f"Image size: {train_obs.shape[2]}x{train_obs.shape[3]}, Channels: {train_obs.shape[4]}")

val_data = np.load('/content/minigrid_data/level1/val.npz', allow_pickle=True)
print(f"\nValidation episodes: {len(val_data['observations'])}")

print("\n✓ Data ready for training!")

# ========================================
# 4. WandB ログイン（初回のみ）
# ========================================
# 下記を実行してWandBにログイン
# !wandb login

print("\n" + "=" * 60)
print("Ready to start training!")
print("=" * 60)
print("\nRun the training cell below:")
```

次に、別のセルで学習を実行：

```bash
%%bash
PYTHONPATH=$PWD:$PYTHONPATH python pldm/train.py \
  --configs pldm/configs/minigrid/level1_test.yaml \
  --values \
    data.minigrid_config.data_path=/content/minigrid_data/level1/train \
    data.minigrid_config.val_path=/content/minigrid_data/level1/val.npz \
    output_root=/content/drive/MyDrive/weblab/world_model_2025/final_research/output/ \
    wandb=true
```

---

## トラブルシューティング

### エラー: "EOFError: No data left in file" または observations.npyが0バイト

**原因1**: データをGoogle Drive上で直接マージした場合、mmap書き込みが正しく完了しない

**解決策1**: ワーカーファイル（worker_*.npz）をローカルにコピーして再マージ
```python
import shutil
import glob
from pathlib import Path

# ワーカーファイルをローカルにコピー
source_workers = "/content/drive/MyDrive/weblab/world_model_2025/final_research/minigrid_data/level1"
local_workers = Path("/content/minigrid_workers")
local_workers.mkdir(parents=True, exist_ok=True)

print("Copying worker files...")
worker_files = glob.glob(f"{source_workers}/worker_*.npz")
for wf in worker_files:
    shutil.copy(wf, local_workers)
    print(f"  ✓ Copied {Path(wf).name}")

# ローカルで再マージ
print("\nMerging datasets locally...")
!cd /content/drive/MyDrive/weblab/world_model_2025/final_research/PLDM && \
python pldm_envs/minigrid/data_generation/merge_datasets.py \
  --input_pattern "/content/minigrid_workers/worker_*.npz" \
  --output_path /content/minigrid_data/level1/train

print("\n✓ Merge completed on local storage!")
```

**原因2**: データが破損しているか、コピーが完了していない

**解決策2**:
```python
# データを削除して再コピー
import shutil
from pathlib import Path

local_data_dir = Path("/content/minigrid_data/level1")
if local_data_dir.exists():
    shutil.rmtree(local_data_dir)
    print("Removed old data")

# 上記の「データをローカルにコピー」セクションを再実行
```

### エラー: "positions.npy not found"

**原因**: データに位置情報が含まれていない

**解決策**: データを `generate_data.py` または `generate_pldm_data.py` で再生成

### コピーに時間がかかる

大規模データセット（10,000エピソードなど）の場合、コピーに5-10分かかることがあります。

**対策**:
- データをGoogle Drive内で圧縮してからコピー
- より小さなデータセットでまずテスト

### メモリ不足

Colabの無料プランではRAMが制限されています。

**対策**:
```python
# batch_sizeを小さくする
!PYTHONPATH=$PWD:$PYTHONPATH python pldm/train.py \
  --configs pldm/configs/minigrid/level1_test.yaml \
  --values \
    data.minigrid_config.batch_size=8 \
    data.minigrid_config.data_path=/content/minigrid_data/level1/train \
    data.minigrid_config.val_path=/content/minigrid_data/level1/val.npz \
    output_root=/content/drive/MyDrive/weblab/world_model_2025/final_research/output/ \
    wandb=true
```

---

## ベストプラクティス

1. **データは毎回コピー**: Colabセッションが切断されるとローカルデータは消えるため、セッション開始時に毎回コピーが必要

2. **チェックポイントはGoogle Driveに保存**: `output_root` をGoogle Drive内に設定

3. **WandBで進捗モニタリング**: `wandb=true` を設定してWandBで学習を監視

4. **小さなデータでテスト**: まず少量データ（100-1000エピソード）でテストしてから本番実行

5. **GPU確認**:
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## データサイズの見積もり

| エピソード数 | 画像サイズ | ディレクトリサイズ | コピー時間（推定） |
|------------|-----------|------------------|--------------------|
| 100        | 72x72     | ~50 MB           | ~10秒              |
| 1,000      | 72x72     | ~500 MB          | ~1分               |
| 10,000     | 72x72     | ~5 GB            | ~5-10分            |

---

## まとめ

Colab環境でMiniGrid学習を実行する鍵は**データをローカルにコピーすること**です。

```
Google Drive → Colab Local Storage → PLDM Training
(mmap非対応)   (mmap対応)              (高速)
```

このワークフローに従えば、Google Colab環境でも問題なくMiniGridのPLDM学習を実行できます。
