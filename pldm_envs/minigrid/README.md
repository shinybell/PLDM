# MiniGrid Environment for PLDM

MiniGrid環境のPLDM統合実装です。ロングホライズン環境における精度検証と、潜在空間の離散化による精度向上の研究に使用します。

## 概要

- **環境**: MiniGrid Empty-8x8
- **ホライズン**: 200-256ステップ（元論文の91-100ステップを超える）
- **目的**: ナビゲーションタスクにおけるロングホライズン予測の検証

## インストール

```bash
pip install minigrid
pip install opencv-python  # 画像リサイズの高速化（オプション）
```

## ディレクトリ構造

```
pldm_envs/minigrid/
├── README.md                           # このファイル
├── __init__.py
├── enums.py                           # データ構造定義
├── configs/
│   ├── empty_8x8.yaml                 # Empty-8x8設定（ホライズン200）
│   └── empty_8x8_h256.yaml            # Empty-8x8設定（ホライズン256）
├── data/
│   ├── __init__.py
│   └── minigrid_dataset.py            # データセットクラス
└── data_generation/
    └── generate_data.py               # データ生成スクリプト
```

## 使い方

### 1. データ生成

#### デバッグ用（少量データ）

```bash
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 10 \
    --max_steps 200 \
    --resize 64 \
    --output_path data/minigrid/empty_8x8_debug.npz
```

#### 訓練用データ（ホライズン200）

```bash
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 20000 \
    --max_steps 200 \
    --resize 64 \
    --seed 42 \
    --output_path data/minigrid/empty_8x8_train.npz
```

#### 検証用データ（ホライズン200）

```bash
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 10000 \
    --max_steps 200 \
    --resize 64 \
    --seed 100 \
    --output_path data/minigrid/empty_8x8_val.npz
```

#### ホライズン256バージョン

```bash
# 訓練用
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 20000 \
    --max_steps 256 \
    --resize 64 \
    --seed 42 \
    --output_path data/minigrid/empty_8x8_h256_train.npz

# 検証用
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 10000 \
    --max_steps 256 \
    --resize 64 \
    --seed 100 \
    --output_path data/minigrid/empty_8x8_h256_val.npz
```

### 2. 設定ファイルの編集

生成したデータのパスを設定ファイルに記入：

```yaml
# configs/empty_8x8.yaml
data:
  minigrid_config:
    data_path: data/minigrid/empty_8x8_train.npz
    val_path: data/minigrid/empty_8x8_val.npz
```

### 3. モデル学習

```bash
# TODO: PLDMの学習スクリプトが完成したら追加
python train.py --config pldm_envs/minigrid/configs/empty_8x8.yaml
```

## データフォーマット

### 生成されるデータ

- **observations**: `[N_episodes, T, H, W, C]` - RGB画像観測
- **actions**: `[N_episodes, T-1]` - 離散アクション（0-6）
- **rewards**: `[N_episodes, T-1]` - 報酬
- **dones**: `[N_episodes, T-1]` - 終端フラグ

### MiniGridアクション空間

```
0: 左を向く
1: 右を向く
2: 前に進む
3: アイテムを拾う
4: アイテムを落とす
5: トグル/アクティベート
6: 完了（ゴール到達）
```

Empty環境では主に 0（左）、1（右）、2（前進）が使われます。

## 環境の詳細

### MiniGrid-Empty-8x8-v0

- **グリッドサイズ**: 8x8
- **観測**: RGB画像（タイルサイズ8の場合、デフォルトで64x64ピクセル）
- **タスク**: ランダムな初期位置から緑のゴールに到達する
- **デフォルトmax_steps**: 256
- **報酬**: ゴール到達時に `1 - 0.9 * (step_count / max_steps)`

### ホライズンの設定

- **ホライズン200**: 元論文（91-100ステップ）の約2倍
- **ホライズン256**: MiniGridのデフォルト設定

## 研究計画との対応

### Phase 1: ベースライン確立

1. ホライズン200でデータ生成・学習
2. ホライズン256でデータ生成・学習
3. ベースラインPLDMの性能評価

### Phase 2: 離散化手法の実装

1. VQ-VAEの実装とEmpty-8x8での学習
2. FSQの実装とEmpty-8x8での学習
3. アブレーションスタディ

### Phase 3: 評価と分析

1. タスク成功率の比較（ベースライン vs VQ vs FSQ）
2. ホライズン別の精度劣化分析
3. 潜在空間の可視化

## トラブルシューティング

### データ生成が遅い

- `--resize` オプションを使って画像サイズを小さくする
- `opencv-python` をインストールして高速化する

### メモリエラー

- エピソード数を減らす
- データセットの `crop_length` パラメータを設定する
- `quick_debug: true` を設定してメモリマップモードを無効化する

## 参考

- [MiniGrid Documentation](https://minigrid.farama.org/)
- [MiniGrid Empty Environment](https://minigrid.farama.org/environments/minigrid/EmptyEnv/)
- [Gymnasium](https://gymnasium.farama.org/)
