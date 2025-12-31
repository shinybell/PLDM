# MiniGrid Data Generation for PLDM

MiniGrid LongHorizon環境（Level 1-3）のオフラインデータを生成し、PLDM学習に使用するためのスクリプト集です。

## クイックスタート

### 1. 全レベルのデータを一括生成

```bash
# 64x64観測で各レベル1000エピソード生成
bash pldm_envs/minigrid/data_generation/generate_all_levels.sh

# 72x72観測で各レベル5000エピソード生成
bash pldm_envs/minigrid/data_generation/generate_all_levels.sh \
  --n_episodes 5000 \
  --obs_size 72
```

生成されるファイル:
```
data/minigrid/
├── level1_64x64_train.npz
├── level2_64x64_train.npz
└── level3_64x64_train.npz
```

### 2. 個別レベルのデータ生成

```bash
# Level 1のみ（64x64, HWCフォーマット, uint8）
python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 1000 \
  --obs_size 64 \
  --output_path data/minigrid/level1_train.npz

# Level 2（72x72）
python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level2-v0 \
  --n_episodes 1000 \
  --obs_size 72 \
  --output_path data/minigrid/level2_72x72_train.npz

# PyTorch形式（CHW, float32, normalized）
python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 1000 \
  --obs_size 64 \
  --channel_first \
  --normalize \
  --output_path data/minigrid/level1_pytorch.npz
```

## データフォーマット

### 生成されるデータ

各`.npz`ファイルには以下のキーが含まれます：

```python
import numpy as np

data = np.load('data/minigrid/level1_64x64_train.npz', allow_pickle=True)

# エピソードデータ（可変長の場合はobject配列）
observations = data['observations']  # shape: (n_episodes,) dtype=object
                                      # 各要素: (T, 64, 64, 3) uint8
actions = data['actions']            # shape: (n_episodes,) dtype=object
                                      # 各要素: (T-1,) int
rewards = data['rewards']            # shape: (n_episodes,) dtype=object
dones = data['dones']                # shape: (n_episodes,) dtype=object

# 固定長の場合（--pad_length指定時）
observations = data['observations']  # shape: (n_episodes, T, 64, 64, 3) uint8
actions = data['actions']            # shape: (n_episodes, T-1) int
```

### 観測フォーマット

| オプション | 形状 | データ型 | 値範囲 |
|-----------|------|---------|--------|
| デフォルト | (T, 64, 64, 3) | uint8 | [0, 255] |
| `--obs_size 72` | (T, 72, 72, 3) | uint8 | [0, 255] |
| `--channel_first` | (T, 3, 64, 64) | uint8 | [0, 255] |
| `--normalize` | (T, 64, 64, 3) | float32 | [0.0, 1.0] |
| `--channel_first --normalize` | (T, 3, 64, 64) | float32 | [0.0, 1.0] |

## PLDM学習への使用

### ステップ1: データ生成

```bash
# 訓練データ（1000エピソード）
bash pldm_envs/minigrid/data_generation/generate_all_levels.sh \
  --n_episodes 1000 \
  --obs_size 64 \
  --data_dir data/minigrid/train

# 評価データ（100エピソード）
bash pldm_envs/minigrid/data_generation/generate_all_levels.sh \
  --n_episodes 100 \
  --obs_size 64 \
  --seed 12345 \
  --data_dir data/minigrid/val
```

### ステップ2: データセット設定

既存のPLDMデータローダーを参考に、MiniGrid用のデータセットクラスを作成：

```python
# 例: datasets/minigrid_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class MiniGridDataset(Dataset):
    def __init__(self, data_path, context_len=16):
        data = np.load(data_path, allow_pickle=True)
        self.observations = data['observations']  # (N,) object array
        self.actions = data['actions']
        self.context_len = context_len

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]  # (T, H, W, C)
        act = self.actions[idx]       # (T-1,)

        # コンテキスト長に合わせてサンプリング
        # ... (wall/diverse_mazeのデータローダー実装を参考)

        return {
            'observations': torch.from_numpy(obs).float() / 255.0,
            'actions': torch.from_numpy(act).long(),
        }
```

### ステップ3: 学習設定

```python
# 例: configs/minigrid_level1.yaml
data:
  train_path: data/minigrid/train/level1_64x64_train.npz
  val_path: data/minigrid/val/level1_64x64_train.npz
  obs_shape: [3, 64, 64]  # CHW format
  action_dim: 7  # MiniGridのアクション空間

model:
  latent_dim: 256
  hidden_dim: 512
  # ... (他のハイパーパラメータ)

training:
  batch_size: 32
  learning_rate: 1e-4
  max_epochs: 100
```

### ステップ4: 学習実行

```bash
# TwoRoomsと同様の方法で学習
python train.py --config configs/minigrid_level1.yaml
```

## コマンドラインオプション

### generate_pldm_data.py

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--env_name` | MiniGrid-LongHorizon-Level1-v0 | 環境名 |
| `--n_episodes` | 1000 | 生成エピソード数 |
| `--obs_size` | 64 | 観測サイズ（64, 72, 84など） |
| `--channel_first` | False | CHWフォーマットを使用 |
| `--normalize` | False | [0,1]に正規化 |
| `--max_steps` | None | 最大ステップ数（環境デフォルト） |
| `--pad_length` | None | 固定長にパディング |
| `--seed` | 42 | ランダムシード |
| `--output_path` | 必須 | 出力ファイルパス |

### generate_all_levels.sh

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--n_episodes` | 1000 | レベルあたりのエピソード数 |
| `--obs_size` | 64 | 観測サイズ |
| `--seed` | 42 | ランダムシード |
| `--data_dir` | data/minigrid | データ出力ディレクトリ |

## トラブルシューティング

### メモリ不足エラー

大量のエピソード（10000+）を生成する場合、メモリ使用量が大きくなります：

```bash
# バッチ処理で分割
for i in {0..9}; do
  python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
    --env_name MiniGrid-LongHorizon-Level1-v0 \
    --n_episodes 1000 \
    --seed $((42 + i * 1000)) \
    --output_path data/minigrid/level1_batch_${i}.npz
done

# 後で結合
python -c "
import numpy as np
batches = [np.load(f'data/minigrid/level1_batch_{i}.npz', allow_pickle=True) for i in range(10)]
combined = {
    'observations': np.concatenate([b['observations'] for b in batches]),
    'actions': np.concatenate([b['actions'] for b in batches]),
    'rewards': np.concatenate([b['rewards'] for b in batches]),
    'dones': np.concatenate([b['dones'] for b in batches]),
}
np.savez_compressed('data/minigrid/level1_full.npz', **combined)
"
```

### 生成速度が遅い

- `--obs_size`を小さくする（64 → 32）
- エピソード数を減らす
- 並列化を検討（複数プロセス）

## 次のステップ

1. ✅ データ生成完了
2. データセットクラスの実装（wallやdiverseの実装を参考）
3. 学習設定ファイルの作成
4. 学習実行
5. 評価・可視化

詳細は[PLDM本体のドキュメント](../../README.md)を参照してください。
