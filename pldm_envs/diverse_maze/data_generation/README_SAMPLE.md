# Diverse Maze サンプルデータ生成ガイド

10エピソードの小規模サンプルデータを生成する手順です。

## クイックスタート

### 方法1: シェルスクリプトで一括実行（推奨）

```bash
# プロジェクトルートから実行
bash pldm_envs/diverse_maze/data_generation/generate_sample_data.sh
```

これで以下のステップが自動実行されます：
1. プロプリオセプティブデータ（位置、速度、アクション）の生成
2. PNG画像のレンダリング
3. Numpy配列への変換

### 方法2: 各ステップを個別に実行

#### Step 1: プロプリオセプティブデータの生成

```bash
python pldm_envs/diverse_maze/data_generation/generate_data.py \
    --output_path pldm_envs/diverse_maze/data/sample_10episodes \
    --config pldm_envs/diverse_maze/configs/sample_10episodes.yaml
```

**生成されるファイル:**
- `data.p` - エピソードデータ（観測値、アクション）
- `metadata.pt` - 設定情報
- `train_maps.pt` - マップレイアウト

#### Step 2: PNG画像のレンダリング

```bash
python pldm_envs/diverse_maze/data_generation/render_data.py \
    --data_path pldm_envs/diverse_maze/data/sample_10episodes
```

**生成されるファイル:**
- `images/` - エピソードごとのPNG画像（トップダウンビュー）

#### Step 3: Numpy配列への変換

```bash
python pldm_envs/diverse_maze/data_generation/postprocess_images.py \
    --data_path pldm_envs/diverse_maze/data/sample_10episodes
```

**生成されるファイル:**
- `images.zarr` - Zarr形式の圧縮画像配列
- `images.npy` - **最終的なNumpy配列ファイル** ← これを使用

## 設定のカスタマイズ

[sample_10episodes.yaml](../configs/sample_10episodes.yaml)を編集することで、以下を変更できます：

```yaml
env: maze2d_small_diverse      # 環境名
num_blocks_width_in_img: 8     # マップサイズ（8x8ブロック）
n_episodes: 10                  # エピソード数
train_maps_n: 1                 # マップ数（1つのマップで10エピソード）
episode_length: 50              # 最大ステップ数
img_size: [64, 64, 3]          # 画像サイズ
sampling_mode: uniform          # サンプリングモード
```

### 利用可能な環境

- `maze2d_small_diverse` - 8x8の小さな迷路（推奨）
- `maze2d_medium_diverse` - より大きな迷路

### エピソード数とマップ数の調整

```yaml
# パターン1: 1つのマップで10エピソード
n_episodes: 10
train_maps_n: 1

# パターン2: 2つのマップで各5エピソード（合計10エピソード）
n_episodes: 5
train_maps_n: 2

# パターン3: 5つのマップで各2エピソード（合計10エピソード）
n_episodes: 2
train_maps_n: 5
```

## 生成されるデータ構造

### data.p (プロプリオセプティブデータ)

```python
import torch
data = torch.load("pldm_envs/diverse_maze/data/sample_10episodes/data.p")

# data is a list of episodes
episode = data[0]  # 1エピソード目
# episode = {
#     "observations": np.array,  # [T, obs_dim] 観測値（位置、速度）
#     "actions": np.array,       # [T-1, 2] アクション系列
#     "map_idx": int,            # マップインデックス
# }
```

### train_maps.pt (マップレイアウト)

```python
import torch
maps = torch.load("pldm_envs/diverse_maze/data/sample_10episodes/train_maps.pt")

# maps = {
#     0: "########\\#OOOOOO#\\...",  # マップ0のレイアウト文字列
#     # O = 通路、# = 壁
# }
```

### images.npy (画像データ)

```python
import numpy as np
images = np.load("pldm_envs/diverse_maze/data/sample_10episodes/images.npy")

# images.shape = [total_frames, H, W, 3]
# 10エピソード × ~50フレーム = ~500フレーム
# H, W = 64 (config.img_size)
```

## 生成時間の目安

**サンプル設定（10エピソード、1マップ、50ステップ）:**
- Step 1（データ生成）: ~30秒
- Step 2（画像レンダリング）: ~2分
- Step 3（Numpy変換）: ~30秒

**合計: 約3分**

## トラブルシューティング

### ImportError: d4rl, mujoco関連

```bash
pip install d4rl
pip install mujoco mujoco-py
```

### 画像が生成されない

1. `data.p`、`metadata.pt`、`train_maps.pt`が存在することを確認
2. `render_data.py`の実行ログを確認

### メモリ不足エラー

- `n_episodes`を減らす（例: 5エピソード）
- `episode_length`を減らす（例: 30ステップ）
- `img_size`を小さくする（例: [32, 32, 3]）

### マップ生成が失敗する

sparsity（壁の密度）を調整：
```yaml
sparsity_low: 50   # 壁が多すぎる場合は下げる
sparsity_high: 75  # 壁が少なすぎる場合は上げる
```

## より大規模なデータセット生成

元の論文の設定（2000エピソード、5マップ）を生成する場合：

```bash
# 既存の5maps設定を使用
python pldm_envs/diverse_maze/data_generation/generate_data.py \
    --output_path pldm_envs/diverse_maze/data/5maps_full \
    --config pldm_envs/diverse_maze/configs/5maps.yaml

# 以降、render_data.py と postprocess_images.py を実行
```

**注意:** フルサイズのデータセットは生成に数時間かかります。

## データの可視化

生成されたマップとエピソードを確認：

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 画像を読み込み
images = np.load("pldm_envs/diverse_maze/data/sample_10episodes/images.npy")

# 最初のエピソードの最初のフレームを表示
plt.imshow(images[0])
plt.title("First frame of first episode")
plt.axis('off')
plt.show()

# エピソード全体をGIFアニメーションとして保存
from PIL import Image
frames = [Image.fromarray(img) for img in images[:51]]  # 最初の51フレーム
frames[0].save(
    "episode_0.gif",
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)
```

## 次のステップ

生成したデータを使って学習する：

```bash
# トレーニング設定でdata_pathを指定
python pldm/train.py --config pldm/configs/diverse_maze/your_config.yaml
```

設定ファイルで以下を指定：
```yaml
data:
  dataset_type: offline_diverse_maze
  offline_diverse_maze_config:
    data_path: pldm_envs/diverse_maze/data/sample_10episodes
```

詳細は元のREADME（[pldm_envs/diverse_maze/README.md](../README.md)）を参照してください。
