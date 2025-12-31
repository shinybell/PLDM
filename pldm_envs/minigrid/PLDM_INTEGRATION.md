# MiniGrid PLDM Integration Guide

MiniGrid環境がPLDMの訓練システムに統合されました。このガイドでは、MiniGrid環境でPLDMを訓練する方法を説明します。

## 概要

以下のコンポーネントが実装されています:

1. **データセットクラス** (`pldm_envs/minigrid/data/minigrid_dataset.py`)
   - `MiniGridDataset`: オフラインデータをロードするPyTorchデータセット
   - `MiniGridSample`: データサンプルの構造
   - `MiniGridDatasetConfig`: データセット設定
   - `minigrid_collate_fn`: DataLoader用のcollate関数

2. **PLDM統合** (`pldm/data/`)
   - `DatasetType.MiniGrid`: 新しいデータセットタイプ
   - `DataConfig.minigrid_config`: MiniGrid設定フィールド
   - `DatasetFactory._create_minigrid_datasets()`: データセット作成メソッド

## クイックスタート

### 1. データ生成

まず、MiniGrid環境からオフラインデータを生成します:

```bash
# Level 1データを生成（72x72画像）
python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 1000 \
  --obs_size 72 \
  --output_path data/minigrid/level1_72x72_train.npz

# Level 2データを生成
python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level2-v0 \
  --n_episodes 1000 \
  --obs_size 72 \
  --output_path data/minigrid/level2_72x72_train.npz
```

### 2. データセットのテスト

データセットクラスが正しく動作するか確認:

```bash
source .venv/bin/activate
PYTHONPATH=/path/to/PLDM:$PYTHONPATH python test_minigrid_dataset_only.py
```

### 3. PLDM設定

PLDM学習用の設定を作成します。Pythonスクリプトまたはコンフィグファイルで設定できます:

```python
from pldm.data.enums import DataConfig, DatasetType
from pldm_envs.minigrid.enums import MiniGridDatasetConfig

# MiniGrid設定
minigrid_config = MiniGridDatasetConfig(
    data_path="data/minigrid/level1_72x72_train.npz",
    val_path="data/minigrid/level1_72x72_val.npz",  # オプション
    sample_length=16,       # コンテキスト長
    img_size=72,            # 画像サイズ
    normalize_images=True,  # [0, 1]に正規化
    batch_size=32,
    include_rewards=False,  # 報酬を使わない場合
    include_dones=False,    # 終端フラグを使わない場合
    crop_length=None,       # 全データを使用
    train=True,
)

# PLDM DataConfig
data_config = DataConfig(
    dataset_type=DatasetType.MiniGrid,
    minigrid_config=minigrid_config,
    normalize=False,  # 画像は既に正規化されている
    num_workers=4,
)
```

### 4. データセット作成

`DatasetFactory`を使ってデータセットを作成:

```python
from pldm.data.dataset_factory import DatasetFactory

factory = DatasetFactory(config=data_config)
datasets = factory.create_datasets()

# 訓練データローダー
train_loader = datasets.ds

# 検証データローダー（オプション）
val_loader = datasets.val_ds

# 使用例
for batch in train_loader:
    states = batch.states    # [B, T, C, H, W]
    actions = batch.actions  # [B, T-1, 1]
    # PLDMの訓練コード...
```

## データフォーマット

### MiniGridSample

各サンプルは以下の構造を持ちます:

```python
@dataclass
class MiniGridSample:
    states: torch.Tensor    # [T, C, H, W] float32, 範囲[0, 1]
    actions: torch.Tensor   # [T-1, 1] int64
    rewards: Optional[torch.Tensor] = None   # [T-1] float32
    dones: Optional[torch.Tensor] = None     # [T-1] bool
```

### バッチ化後

DataLoaderを通すと、バッチ次元が追加されます:

```python
batch.states:  [B, T, C, H, W]  # B=バッチサイズ, T=時系列長
batch.actions: [B, T-1, 1]
batch.rewards: [B, T-1] or None
batch.dones:   [B, T-1] or None
```

## 設定パラメータ

### MiniGridDatasetConfig

| パラメータ | 型 | デフォルト | 説明 |
|----------|-----|-----------|------|
| `data_path` | str | (必須) | 訓練データのパス (.npz) |
| `val_path` | Optional[str] | None | 検証データのパス (.npz) |
| `sample_length` | int | 16 | サンプルの時系列長 |
| `img_size` | int | 64 | 画像サイズ（64 or 72推奨） |
| `normalize_images` | bool | True | [0, 1]に正規化 |
| `batch_size` | int | 32 | バッチサイズ |
| `include_rewards` | bool | False | 報酬を含める |
| `include_dones` | bool | False | 終端フラグを含める |
| `crop_length` | Optional[int] | None | データセット長を制限 |
| `train` | bool | True | 訓練モード |
| `quick_debug` | bool | False | デバッグモード |

## トラブルシューティング

### データが見つからない

```
FileNotFoundError: data/minigrid/level1_72x72_train.npz
```

→ `generate_pldm_data.py`を使ってデータを生成してください。

### メモリエラー

大量のデータを読み込む場合、メモリ不足になる可能性があります:

- `crop_length`を設定してデータセットを小さくする
- `batch_size`を小さくする
- `num_workers`を調整する

### 画像サイズの不一致

```
RuntimeError: size mismatch
```

→ データ生成時の`--obs_size`とconfig の`img_size`が一致しているか確認してください。

## 次のステップ

1. **PLDM訓練の実行**: 既存のPLDM訓練スクリプトにMiniGrid設定を渡す
2. **モデル評価**: 訓練したモデルでMiniGrid環境での推論を実行
3. **ハイパーパラメータ調整**: `sample_length`, `batch_size`などを調整

## 参考資料

- MiniGrid環境のREADME: `pldm_envs/minigrid/README.md`
- データ生成ガイド: `pldm_envs/minigrid/QUICKSTART_LONG_HORIZON.md`
- カスタムマップガイド: `pldm_envs/minigrid/CUSTOM_MAP_GUIDE.md`
- Atari統合の実装: `pldm/data/dataset_factory.py` (参考)

## 実装の詳細

### ファイル構造

```
pldm_envs/minigrid/
├── data/
│   ├── __init__.py                 # MiniGridDataset, collate_fnのエクスポート
│   └── minigrid_dataset.py         # データセットクラスの実装
├── enums.py                        # MiniGridDatasetConfigの再エクスポート
└── data_generation/
    └── generate_pldm_data.py       # データ生成スクリプト

pldm/data/
├── enums.py                        # DatasetType.MiniGrid, DataConfig
└── dataset_factory.py              # _create_minigrid_datasets()
```

### 変更履歴

**2026-01-01**: MiniGrid PLDM統合完了
- `MiniGridDataset`クラスの実装
- `DatasetFactory`への統合
- `MiniGridDatasetConfig`の統合
- テストスクリプトの作成と検証

---

質問や問題がある場合は、GitHubのIssueを作成してください。
