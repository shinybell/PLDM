# MiniGrid サンプルデータ生成ガイド

10エピソードの小規模サンプルデータを生成する手順です。

## クイックスタート

### 方法1: シェルスクリプトで一括実行（推奨）

```bash
# プロジェクトルートから実行
bash pldm_envs/minigrid/data_generation/generate_sample_data.sh
```

これで以下のステップが自動実行されます：
1. プロプリオセプティブデータ（位置、アクション）の生成
2. PNG画像のレンダリング
3. Numpy配列への変換

### 方法2: 各ステップを個別に実行

#### Step 1: プロプリオセプティブデータの生成

```bash
python pldm_envs/minigrid/data_generation/generate_data_from_config.py \
    --config pldm_envs/minigrid/configs/sample_10episodes.yaml \
    --output_path pldm_envs/minigrid/data/sample_10episodes
```

**生成されるファイル:**
- `data.p` - エピソードデータ（位置、アクション、報酬など）
- `metadata.pt` - 設定情報

#### Step 2: PNG画像のレンダリング

```bash
python pldm_envs/minigrid/data_generation/render_data.py \
    --data_path pldm_envs/minigrid/data/sample_10episodes
```

**生成されるファイル:**
- `images/` - エピソードごとのPNG画像

#### Step 3: Numpy配列への変換

```bash
python pldm_envs/minigrid/data_generation/postprocess_images.py \
    --data_path pldm_envs/minigrid/data/sample_10episodes
```

**生成されるファイル:**
- `images.zarr` - Zarr形式の圧縮画像配列
- `images.npy` - **最終的なNumpy配列ファイル** ← これを使用

## 設定のカスタマイズ

[sample_10episodes.yaml](../configs/sample_10episodes.yaml)を編集することで、以下を変更できます：

```yaml
env_name: MiniGrid-Empty-8x8-v0  # 環境名
n_episodes: 10                    # エピソード数
episode_length: 100               # 最大ステップ数
img_size: [64, 64, 3]            # 画像サイズ
tile_size: 8                      # タイルサイズ
seed: 42                          # ランダムシード
```

### 利用可能な環境

- `MiniGrid-Empty-8x8-v0` - 8x8の空環境
- `MiniGrid-Empty-16x16-v0` - 16x16の空環境
- `MiniGrid-FourRooms-v0` - 4部屋の迷路
- その他のMiniGrid環境

## 生成されるデータ構造

### data.p (プロプリオセプティブデータ)

```python
import torch
data = torch.load("pldm_envs/minigrid/data/sample_10episodes/data.p")

# data is a list of episodes
episode = data[0]  # 1エピソード目
# episode = {
#     "actions": np.array,      # [T-1] アクション系列
#     "rewards": np.array,      # [T-1] 報酬系列
#     "dones": np.array,        # [T-1] 終了フラグ
#     "positions": np.array,    # [T, 2] エージェント位置 (x, y)
#     "directions": np.array,   # [T] エージェント向き (0-3)
# }
```

### images.npy (画像データ)

```python
import numpy as np
images = np.load("pldm_envs/minigrid/data/sample_10episodes/images.npy")

# images.shape = [total_frames, H, W, 3]
# 10エピソード × ~100フレーム = ~1000フレーム
# H, W = 64 (config.img_size)
```

## トラブルシューティング

### ImportError: No module named 'minigrid'

```bash
pip install minigrid
```

### 画像が生成されない

1. `data.p`と`metadata.pt`が存在することを確認
2. `render_data.py`の実行ログを確認

### メモリ不足エラー

- `n_episodes`を減らす（例: 5エピソード）
- `img_size`を小さくする（例: [32, 32, 3]）

## より大規模なデータセット生成

1000エピソードなど大規模データを生成する場合：

```yaml
# configs/large_dataset.yaml を作成
env_name: MiniGrid-Empty-8x8-v0
n_episodes: 1000
episode_length: 200
img_size: [64, 64, 3]
```

```bash
python pldm_envs/minigrid/data_generation/generate_data_from_config.py \
    --config pldm_envs/minigrid/configs/large_dataset.yaml \
    --output_path pldm_envs/minigrid/data/large_dataset

# 以降、render_data.py と postprocess_images.py を実行
```

## 次のステップ

生成したデータを使って学習する：

```bash
python pldm/train.py --config pldm/configs/minigrid/your_config.yaml
```

詳細は[PLDM統合ガイド](../PLDM_INTEGRATION.md)を参照してください。
