# MiniGrid環境 完全ガイド - データ生成から学習まで

**対象読者**: PLDMの元実装(DiverseMaze/TwoRooms)は知っているが、MiniGrid環境は初めての方

このガイドでは、MiniGrid環境でのPLDM学習の全プロセスを、順を追って説明します。

---

## 目次

1. [MiniGridとは](#1-minigridとは)
2. [環境セットアップ](#2-環境セットアップ)
3. [データ生成](#3-データ生成)
4. [PLDM学習](#4-pldm学習)
5. [評価とMPC](#5-評価とmpc)
6. [トラブルシューティング](#6-トラブルシューティング)

---

## 1. MiniGridとは

### 概要

MiniGridは、グリッドベースの2D環境です。DiverseMaze/TwoRoomsとの主な違い:

| 項目 | DiverseMaze/TwoRooms | MiniGrid |
|------|---------------------|----------|
| **環境タイプ** | 連続2D迷路 | グリッドワールド |
| **観測** | 64x64 RGB鳥瞰図 | 72x72 RGB鳥瞰図（設定可能） |
| **アクション** | 連続（2D速度） | 離散（7種類） |
| **レンダリング** | MuJoCo | Gymnasium |
| **データ生成** | 3ステップ（状態→画像→NPY） | 1ステップ（直接NPZ） |

### MiniGridのアクション空間

```python
0: 左を向く (turn left)
1: 右を向く (turn right)
2: 前進 (move forward)
3: オブジェクトを拾う (pick up)
4: オブジェクトを置く (drop)
5: トグル/アクティベート (toggle/activate)
6: 完了 (done)
```

### Long Horizon環境（3レベル）

本プロジェクトでは、長期計画が必要な3つのレベルを用意:

```
Level 1 (12x12): ドア1つ、鍵1つ
Level 2 (16x16): ドア2つ、鍵2つ
Level 3 (20x20): ドア3つ、鍵3つ、複雑な迷路構造
```

---

## 2. 環境セットアップ

### 必要なパッケージ

```bash
# 仮想環境をアクティベート
source .venv/bin/activate

# MiniGridのインストール確認
python -c "import minigrid; print('MiniGrid OK')"
python -c "import gymnasium; print('Gymnasium OK')"
python -c "import pldm_envs.minigrid; print('MiniGrid envs registered')"
```

### ディレクトリ構造

```
PLDM/
├── pldm_envs/
│   └── minigrid/
│       ├── __init__.py              # カスタム環境の登録
│       ├── envs/                    # Level 1-3環境定義
│       ├── wrappers.py              # PLDMWrapper（画像変換）
│       ├── data_generation/         # データ生成スクリプト
│       │   ├── generate_pldm_data.py      # 基本データ生成
│       │   ├── generate_data.py           # 汎用データ生成
│       │   ├── parallel_generate.sh       # 並列データ生成
│       │   └── merge_datasets.py          # データセット結合
│       └── configs/                 # 環境設定YAML
├── pldm/
│   ├── planning/
│   │   └── minigrid/
│   │       └── mpc.py               # MPC評価
│   └── configs/
│       └── minigrid/                # 学習設定（今後追加）
└── data/
    └── minigrid/                    # 生成データ保存先
```

---

## 3. データ生成

### 3-1. クイックスタート（少量データでテスト）

まず10エピソードで動作確認:

```bash
# venv環境をアクティベート
source .venv/bin/activate

# Level 1で10エピソード生成（72x72）
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 10 \
  --obs_size 72 \
  --output_path data/minigrid/test_level1.npz

# データ確認
python -c "
import numpy as np
data = np.load('data/minigrid/test_level1.npz', allow_pickle=True)
print(f'Episodes: {len(data[\"observations\"])}')
print(f'Obs shape: {data[\"observations\"][0].shape}')
print(f'Obs dtype: {data[\"observations\"][0].dtype}')
print(f'Value range: [{data[\"observations\"][0].min()}, {data[\"observations\"][0].max()}]')
"
```

**期待される出力:**
```
Episodes: 10
Obs shape: (T, 72, 72, 3)  # T=エピソードステップ数（可変）
Obs dtype: uint8
Value range: [0, 255]
```

### 3-2. 本格的なデータ生成（シングルプロセス）

#### 方法A: PLDM専用スクリプト（推奨）

PLDMWrapperを使って、PLDM学習に最適化された形式で生成:

```bash
# Level 1: 1000エピソード、72x72
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 1000 \
  --obs_size 72 \
  --output_path data/minigrid/level1_train.npz

# Level 2: 1000エピソード、72x72
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level2-v0 \
  --n_episodes 1000 \
  --obs_size 72 \
  --output_path data/minigrid/level2_train.npz

# Level 3: 1000エピソード、72x72
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level3-v0 \
  --n_episodes 1000 \
  --obs_size 72 \
  --output_path data/minigrid/level3_train.npz
```

**オプション:**
- `--obs_size 64`: 64x64画像（DiverseMazeと同じ）
- `--channel_first`: PyTorch形式 (C, H, W)
- `--normalize`: [0, 1]に正規化（float32）
- `--pad_length 201`: 全エピソードを201ステップに固定

#### 方法B: 汎用スクリプト

より柔軟な設定が可能:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_data.py \
  --env_name MiniGrid-Empty-8x8-v0 \
  --n_episodes 1000 \
  --output_path data/minigrid/empty_8x8_train.npz \
  --resize 72 \
  --max_steps 200
```

### 3-3. 並列データ生成（高速化）

**重要**: 大量データ（10,000エピソード以上）を生成する場合は並列化を推奨

#### 並列化の効果

| ワーカー数 | 10,000エピソードの所要時間 | 高速化率 |
|-----------|-------------------------|---------|
| 1（シングル） | 約30分 | 1x（ベースライン） |
| 4 | 約8分 | 3.75x |
| 8 | 約4分 | 7.5x |

#### 使い方

```bash
# venv環境をアクティベート
source .venv/bin/activate

# 基本的な使い方: 4ワーカーで10,000エピソード生成
bash pldm_envs/minigrid/data_generation/parallel_generate.sh \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 10000 \
  --workers 4 \
  --output_dir data/minigrid/level1_10k \
  --max_steps 300 \
  --resize 72

# 高度な使い方: 20分割を8並列で実行（メモリ効率重視）
bash pldm_envs/minigrid/data_generation/parallel_generate.sh \
  --env_name MiniGrid-LongHorizon-Level2-v0 \
  --n_episodes 20000 \
  --workers 20 \
  --jobs 8 \
  --output_dir data/minigrid/level2_20k \
  --resize 72
```

**パラメータ説明:**
- `--workers`: データを何分割するか（n_episodesはこれで割り切れる必要がある）
- `--jobs`: 同時に実行するワーカー数（省略時はworkersと同じ）
- `--output_dir`: 出力ディレクトリ（`train/`以下にnpyファイルが生成される）

**並列生成の仕組み:**

```
1. データを分割（例: 10,000エピソード → 4ワーカー = 各2,500エピソード）
2. 各ワーカーが並列実行
   Worker 0: episodes 0-2499     → worker_0.npz
   Worker 1: episodes 2500-4999  → worker_1.npz
   Worker 2: episodes 5000-7499  → worker_2.npz
   Worker 3: episodes 7500-9999  → worker_3.npz
3. 自動結合
   merge_datasets.py → data/minigrid/level1_10k/train/
4. クリーンアップ（オプション）
   worker_*.npz を削除
```

**進捗確認:**

別のターミナルで:
```bash
# すべてのワーカーのログを監視
tail -f data/minigrid/level1_10k/worker_*.log

# 特定のワーカーのみ
tail -f data/minigrid/level1_10k/worker_0.log
```

**出力形式:**

並列生成では、メモリ効率のためディレクトリ形式で保存:
```
data/minigrid/level1_10k/
├── train/
│   ├── observations.npy  # メモリマップ可能
│   ├── actions.npy
│   ├── rewards.npy
│   └── dones.npy
└── worker_*.log  # ログファイル
```

### 3-4. データフォーマット

#### シングルプロセス生成（.npz）

```python
{
    'observations': (N,) of [(T_i, H, W, C), ...]  # N=エピソード数、T_i=各エピソードのステップ数（可変）
    'actions': (N,) of [(T_i-1,), ...]             # 離散アクション 0-6
    'rewards': (N,) of [(T_i-1,), ...]             # 報酬（通常0、ゴール時1）
    'dones': (N,) of [(T_i-1,), ...]               # 終了フラグ
}
```

**注意**: DiverseMazeと異なり、`positions`は含まれません（MiniGridでは不要）

#### 並列生成（ディレクトリ）

```python
train/
├── observations.npy: (N*T, H, W, C)  # 全エピソードを平坦化
├── actions.npy: (N*T,)
├── rewards.npy: (N*T,)
└── dones.npy: (N*T,)
```

メモリマップで読み込み可能（大規模データ向け）:
```python
import numpy as np
observations = np.load('data/minigrid/level1_10k/train/observations.npy', mmap_mode='r')
```

### 3-5. データ確認と可視化

```python
import numpy as np
import matplotlib.pyplot as plt

# シングルプロセス生成の場合
data = np.load('data/minigrid/level1_train.npz', allow_pickle=True)

print('=== Dataset Info ===')
print(f'Episodes: {len(data["observations"])}')
print(f'First episode steps: {len(data["observations"][0])}')
print(f'Observation shape: {data["observations"][0].shape}')
print(f'Value range: [{data["observations"][0].min()}, {data["observations"][0].max()}]')

# 最初のエピソードの最初の4フレームを可視化
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    axes[i].imshow(data['observations'][0][i])
    axes[i].set_title(f'Frame {i}')
    axes[i].axis('off')
plt.savefig('minigrid_sample.png', dpi=150, bbox_inches='tight')
print('Saved: minigrid_sample.png')

# 並列生成の場合
obs = np.load('data/minigrid/level1_10k/train/observations.npy', mmap_mode='r')
print(f'\nParallel dataset shape: {obs.shape}')
```

---

## 4. PLDM学習

### 4-1. データセットクラス（既に実装済み✅）

MiniGrid用のデータセットクラスは既に実装されています。

**実装ファイル:** [pldm_envs/minigrid/data/minigrid_dataset.py](pldm_envs/minigrid/data/minigrid_dataset.py)

**主な機能:**

```python
from pldm_envs.minigrid.data.minigrid_dataset import (
    MiniGridDataset,
    MiniGridDatasetConfig,
    MiniGridSample
)

# データセット設定
config = MiniGridDatasetConfig(
    data_path='data/minigrid/level1_train.npz',  # または train/ ディレクトリ
    val_path='data/minigrid/level1_val.npz',
    sample_length=16,      # シーケンス長（コンテキスト）
    img_size=72,           # 画像サイズ
    normalize_images=True, # [0, 1]に正規化
    batch_size=32,
)

# データセット作成
dataset = MiniGridDataset(config)

# サンプル取得
sample = dataset[0]  # MiniGridSample
# sample.states: [T, C, H, W] torch.Tensor
# sample.actions: [T-1, NUM_ACTIONS] torch.Tensor (one-hot)
# sample.locations: [T, 2] torch.Tensor (x, y)
```

**重要な特徴:**

1. **位置情報が必要**: データには`positions`フィールドが必須
   - データ生成時に`generate_data.py`または`generate_pldm_data.py`を使用
   - 両方とも自動的に位置情報を保存

2. **2つの形式に対応**:
   - `.npz`ファイル（シングルプロセス生成）
   - ディレクトリ（並列生成、メモリマップ可能）

3. **可変長エピソード対応**: パディングなしのデータもサポート

4. **自動リサイズ**: 異なる画像サイズのデータも読み込み可能

### 4-2. 設定ファイル（既に実装済み✅）

設定ファイルの例が既にあります:

**実装ファイル:** [pldm/configs/minigrid/level1_test.yaml](pldm/configs/minigrid/level1_test.yaml)

**主な設定項目:**

```yaml
# データ設定
data:
  dataset_type: DatasetType.MiniGrid
  minigrid_config:
    data_path: "data/minigrid/level1_train.npz"
    val_path: "data/minigrid/level1_val.npz"
    sample_length: 16       # コンテキスト長
    img_size: 72            # 画像サイズ
    normalize_images: true  # [0, 1]に正規化
    batch_size: 16

# モデル設定（HJEPAアーキテクチャ）
hjepa:
  level1:
    backbone:
      arch: impala          # IMPALA CNN
      channels: 3           # RGB
    predictor:
      predictor_arch: rnnV2 # RNN予測器
    action_dim: 7           # MiniGrid 7アクション

# 損失関数
objectives_l1:
  objectives:
    - VICReg  # 表現学習
    - IDM     # 逆動力学モデル

# 学習設定
epochs: 100
base_lr: 0.001
optimizer_type: Adam

# 評価設定
eval_cfg:
  probing:
    probe_targets: "locations"  # 位置予測でprobing
  minigrid_planning:
    n_envs: 4
    n_steps: 200
    levels: "level1"
```

**設定ファイルをコピーして独自の実験を作成:**

```bash
# テスト設定をコピー
cp pldm/configs/minigrid/level1_test.yaml pldm/configs/minigrid/my_experiment.yaml

# 設定を編集
vim pldm/configs/minigrid/my_experiment.yaml
```

### 4-3. 学習実行

**前提条件:**
1. データが生成済み（`data/minigrid/level1_train.npz`など）
2. venv環境がアクティブ

**基本的な学習実行:**

```bash
# venv環境をアクティベート
source .venv/bin/activate

# 既存の設定で学習開始
python pldm/train.py --configs pldm/configs/minigrid/level1_test.yaml

# WandBを有効化して学習
python pldm/train.py \
  --configs pldm/configs/minigrid/level1_test.yaml \
  --values wandb=True run_project=my-minigrid-project

# デバッグモード（少量データで高速確認）
python pldm/train.py \
  --configs pldm/configs/minigrid/level1_test.yaml \
  --values quick_debug=True epochs=1

# データパスを上書き（--valuesでドット記法で指定）
python pldm/train.py \
  --configs pldm/configs/minigrid/level1_test.yaml \
  --values data.minigrid_config.data_path=data/minigrid/my_data.npz

# 複数の設定を上書き
python pldm/train.py \
  --configs pldm/configs/minigrid/level1_test.yaml \
  --values wandb=True epochs=50 base_lr=0.0005

# 複数GPU使用（実装されている場合）
python pldm/train.py \
  --configs pldm/configs/minigrid/level1_test.yaml \
  --values num_gpus=2
```

**重要**:
- `--configs`（複数形）: YAMLファイルのパス
- `--values`: ドット記法で設定を上書き（例: `data.minigrid_config.batch_size=32`）

**注意事項:**
- 初回実行時はデータの読み込みに時間がかかる場合があります
- `positions`フィールドがないデータはエラーになります
- 並列生成したデータ（ディレクトリ形式）も直接使用可能

### 4-4. 学習の監視

WandBで以下をモニタリング:

- **Reconstruction Loss**: 画像再構成の精度
- **Latent Loss**: 潜在表現の学習
- **Prediction Accuracy**: 将来フレーム予測
- **Action Prediction**: アクション予測精度（MiniGrid特有）

---

## 5. 評価とMPC

### 5-1. MPC評価の実行

学習済みモデルで、Model Predictive Control (MPC) を実行:

```bash
# Level 1でMPC評価
python pldm/planning/minigrid/run_mpc.py \
  --checkpoint_path path/to/checkpoint.pt \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_envs 4 \
  --n_steps 100 \
  --img_size 72

# 複数レベルで評価
for level in 1 2 3; do
  python pldm/planning/minigrid/run_mpc.py \
    --checkpoint_path path/to/checkpoint.pt \
    --env_name MiniGrid-LongHorizon-Level${level}-v0 \
    --n_envs 4 \
    --n_steps 200 \
    --img_size 72
done
```

### 5-2. MPCの仕組み

DiverseMazeと同様の勾配ベース最適化:

```python
1. 現在の観測をエンコード
2. 複数のアクションシーケンスを計画
3. 勾配降下でゴールに近づくアクションを探索
4. 最適なアクションを実行
5. 次のステップで再計画（Receding Horizon）
```

**MiniGrid特有の違い:**
- アクション空間が離散（0-6）だが、計画時は連続緩和
- 最終的に離散化して実行

---

## 6. トラブルシューティング

### 6-1. データ生成時のエラー

#### エラー: `ModuleNotFoundError: No module named 'pldm_envs'`

**原因**: PYTHONPATHが設定されていない

**解決策**:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
# または
PYTHONPATH=$PWD:$PYTHONPATH python ...
```

#### エラー: `gym.error.UnregisteredEnv: Environment MiniGrid-LongHorizon-Level1-v0 doesn't exist`

**原因**: カスタム環境が登録されていない

**解決策**:
```bash
# Pythonで確認
python -c "import pldm_envs.minigrid; print('OK')"
```

#### エラー: 並列生成で `n_episodes must be divisible by workers`

**原因**: エピソード数がワーカー数で割り切れない

**解決策**:
```bash
# 10,007 → 10,000 または 10,008 に調整
--n_episodes 10000 --workers 4  # OK
--n_episodes 10007 --workers 4  # Error
```

### 6-2. 学習時のエラー

#### エラー: `CUDA out of memory`

**解決策**:
- バッチサイズを減らす: `batch_size: 16`
- 画像サイズを減らす: `obs_size: 64`
- 勾配累積を使う: `accumulate_grad_batches: 2`

#### エラー: 学習が収束しない

**チェック項目**:
1. データが正しく読み込めているか
2. 画像が正規化されているか（[0, 1]）
3. アクション空間が正しいか（0-6）
4. 学習率が適切か（3e-4推奨）

### 6-3. MPC評価時のエラー

#### エラー: `final_preds_dist` の計算エラー

**原因**: 画像サイズの不一致（64 vs 72）

**解決策**:
- チェックポイントの学習時の`img_size`と合わせる
- `--img_size 72`オプションを明示的に指定

---

## 付録: DiverseMazeとの比較

### データ生成の違い

| 工程 | DiverseMaze | MiniGrid |
|------|-------------|----------|
| ステップ1 | 状態データ生成 (`generate_data.py`) | データ生成 (`generate_pldm_data.py`) |
| ステップ2 | 画像レンダリング (`render_data.py`) | - |
| ステップ3 | NPY変換 (`postprocess_images.py`) | - |
| 並列化 | ワーカー分割 | ワーカー分割（同じ方式） |

### 観測の違い

```python
# DiverseMaze
observations: (N, T, 64, 64, 3)  # uint8, [0, 255]
positions: (N, T, 2)              # エージェント位置（連続座標）

# MiniGrid
observations: (N, T, 72, 72, 3)  # uint8, [0, 255]
positions: (N, T, 2)              # エージェント位置（グリッド座標）
```

**注意**: 以前は「MiniGridにpositionsは不要」と記載していましたが、**実際にはMiniGridでも`positions`フィールドが必要**です。データセットが位置情報を使ってprobingを行うためです。

### アクション空間の違い

```python
# DiverseMaze
actions: (N, T, 2)  # 連続（vx, vy）

# MiniGrid
actions: (N, T)     # 離散（0-6）
```

---

## まとめ

### 最小限の手順（すぐ実行可能！）

```bash
# 1. 環境確認
source .venv/bin/activate
python -c "import pldm_envs.minigrid; print('OK')"

# 2. テストデータ生成（10エピソード、約1分）
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 10 \
  --obs_size 72 \
  --output_path data/minigrid/level1_test.npz

# 3. データ確認
python -c "
import numpy as np
data = np.load('data/minigrid/level1_test.npz', allow_pickle=True)
print(f'Episodes: {len(data[\"observations\"])}')
print(f'Has positions: {\"positions\" in data}')
print(f'Sample shape: {data[\"observations\"][0].shape}')
"

# 4. 学習実行（テスト設定で3エポック）
python pldm/train.py --configs pldm/configs/minigrid/level1_test.yaml

# ===== 本格的な実験 =====

# 5. 大規模データ生成（並列、10,000エピソード、約8分 @4workers）
bash pldm_envs/minigrid/data_generation/parallel_generate.sh \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 10000 \
  --workers 4 \
  --output_dir data/minigrid/level1_10k \
  --resize 72

# 6. 設定ファイルをコピーして編集
cp pldm/configs/minigrid/level1_test.yaml pldm/configs/minigrid/my_experiment.yaml
# data_pathを data/minigrid/level1_10k/train に変更
# epochs を 100 に変更

# 7. 本格的な学習実行
python pldm/train.py \
  --config pldm/configs/minigrid/my_experiment.yaml \
  --values wandb=True

# 8. MPC評価（学習後）
python pldm/planning/minigrid/run_mpc.py \
  --checkpoint_path checkpoints/my_experiment/best.pt \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_envs 4 \
  --n_steps 200 \
  --img_size 72
```

### 実装状況

1. ✅ データ生成スクリプト実装完了
2. ✅ 並列データ生成実装完了
3. ✅ データセットクラス実装完了
4. ✅ 学習設定ファイル作成完了
5. ✅ MPC評価コード実装完了
6. 🔄 実際の学習実行（ユーザー側で実施）
7. 🔄 結果の評価と可視化（ユーザー側で実施）

**すぐに始められます！** データを生成して学習を開始できます。

---

## 参考ドキュメント

- [README.md](README.md) - MiniGrid環境の基本情報
- [LONG_HORIZON.md](LONG_HORIZON.md) - Long Horizon環境の詳細
- [QUICKSTART_DATA_GENERATION.md](QUICKSTART_DATA_GENERATION.md) - データ生成クイックスタート
- [data_generation/README_PARALLEL.md](data_generation/README_PARALLEL.md) - 並列データ生成の詳細
- [EVALUATION.md](EVALUATION.md) - MPC評価の詳細
- [PLDM_INTEGRATION.md](PLDM_INTEGRATION.md) - PLDM統合ガイド

---

**最終更新**: 2026-01-03
**著者**: PLDM MiniGrid統合チーム
