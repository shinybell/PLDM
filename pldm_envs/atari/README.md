# Atari環境 (MsPacman) のPLDM統合

このディレクトリには、Atari環境（特にMsPacman）をPLDMで使用するための実装が含まれています。

## 概要

Atari 2600ゲーム（MsPacmanなど）の観測データを使って、階層的な表現学習を行います。

**参考**:
- [Gymnasium Atari Documentation](https://gymnasium.farama.org/environments/atari/)
- [ALE-py](https://pypi.org/project/ale-py/)

## インストール

### 1. Gymnasium + Atariのインストール

```bash
# Gymnasium と Atari環境をインストール
pip install 'gymnasium[atari]'

# または個別にインストール
pip install gymnasium
pip install ale-py

# ROMs are now packaged with ale-py>=0.9, so no separate installation needed
```

### 2. 画像処理ライブラリ（オプション）

```bash
# OpenCV (画像リサイズに使用)
pip install opencv-python

# または PyTorch のみでも動作
```

## データ生成

### ランダムポリシーでデータ生成

```bash
# デバッグ用（10エピソード）
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name MsPacman-v5 \
  --n_episodes 10 \
  --output_path data/atari/mspacman_debug.npz \
  --pad_length 100

# 訓練用（1000エピソード）
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name MsPacman-v5 \
  --n_episodes 1000 \
  --output_path data/atari/mspacman_train.npz \
  --pad_length 1000 \
  --seed 42

# 検証用（200エピソード）
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name MsPacman-v5 \
  --n_episodes 200 \
  --output_path data/atari/mspacman_val.npz \
  --pad_length 1000 \
  --seed 43
```

### データフォーマット

生成されるNPZファイルには以下が含まれます:

```python
{
    'observations': np.ndarray,  # [N_episodes, T, H, W, C]
                                 # H=210, W=160, C=3 (RGB) or C=1 (grayscale)
    'actions': np.ndarray,       # [N_episodes, T-1] (離散アクション)
    'rewards': np.ndarray,       # [N_episodes, T-1]
    'dones': np.ndarray,         # [N_episodes, T-1] (bool)
}
```

## データセットのテスト

```bash
# ダミーデータでテスト
python pldm_envs/atari/test_dataset.py

# 実データでテスト
python pldm_envs/atari/test_dataset.py --data_path data/atari/mspacman_debug.npz
```

## 訓練

### 設定ファイルの編集

[configs/mspacman.yaml](configs/mspacman.yaml) を編集して、データパスを設定:

```yaml
data:
  atari_config:
    data_path: /path/to/mspacman_train.npz
    val_path: /path/to/mspacman_val.npz
```

### 訓練の実行

```bash
# 基本的な訓練
python pldm/train.py --configs pldm_envs/atari/configs/mspacman.yaml

# クイックデバッグ
python pldm/train.py \
  --configs pldm_envs/atari/configs/mspacman.yaml \
  --values \
    quick_debug=true \
    epochs=1 \
    data.atari_config.crop_length=100

# パラメータをコマンドラインで上書き
python pldm/train.py \
  --configs pldm_envs/atari/configs/mspacman.yaml \
  --values \
    data.atari_config.batch_size=64 \
    base_lr=0.0002 \
    epochs=200
```

## ディレクトリ構造

```
pldm_envs/atari/
├── __init__.py
├── README.md                      # このファイル
├── enums.py                       # データ構造と設定の定義
├── data/
│   ├── __init__.py
│   └── atari_dataset.py          # データセットクラス
├── data_generation/
│   └── generate_data.py          # データ生成スクリプト
├── evaluation/
│   └── (今後追加予定)
├── configs/
│   └── mspacman.yaml             # MsPacman設定ファイル
└── test_dataset.py               # テストスクリプト
```

## 実装の詳細

### データセットクラス

[data/atari_dataset.py](data/atari_dataset.py) には以下が実装されています:

- **AtariDataset**: オフラインデータ（NPZ/PKL）からデータをロード
- **AtariOnlineDataset**: Gymnasiumから直接データを生成（未完成）

### 主要な機能

1. **画像の前処理**:
   - リサイズ (210x160 → 64x64)
   - グレースケール変換（オプション）
   - 正規化 (0-255 → 0-1)
   - フレームスタック（オプション）

2. **スライディングウィンドウ**:
   - 長いエピソードから複数のサンプルを抽出
   - `sample_length` で指定した長さのサブシーケンスを生成

3. **メモリ効率**:
   - `mmap_mode='r'` で大容量データを効率的に扱う
   - `quick_debug=True` でメモリに全ロード（小規模データ用）

## 設定パラメータ

### AtariDatasetConfig

主要なパラメータ:

- **env_name**: Atari環境名 (例: "MsPacman-v5", "Pong-v5")
- **data_path**: データファイルのパス
- **batch_size**: バッチサイズ
- **sample_length**: サンプルのタイムステップ数
- **img_size**: リサイズ後の画像サイズ (デフォルト: 64)
- **grayscale**: グレースケール変換するか (デフォルト: False)
- **frame_stack**: フレームスタック数 (デフォルト: 1)
- **normalize_images**: 画像を0-1に正規化するか (デフォルト: True)

## 他のAtariゲームへの適用

MsPacman以外のゲームも同様に使用できます:

```yaml
# Pong
data:
  atari_config:
    env_name: "Pong-v5"
    # ... other settings ...

# Breakout
data:
  atari_config:
    env_name: "Breakout-v5"
    # ... other settings ...
```

利用可能な環境:
- `MsPacman-v5` (9 actions)
- `Pong-v5` (6 actions)
- `Breakout-v5` (4 actions)
- `SpaceInvaders-v5` (6 actions)
- その他、[Gymnasium Atari環境一覧](https://gymnasium.farama.org/environments/atari/)を参照

**注意**: アクション数が異なる場合は、設定ファイルの `hjepa.level1.action_dim` を変更してください。

## トラブルシューティング

### Import Error: gymnasium

```bash
pip install 'gymnasium[atari]'
```

### ROMs not found

ALE-py 0.9以降では、ROMsは自動的にパッケージに含まれます。古いバージョンを使用している場合:

```bash
pip install --upgrade ale-py
```

### メモリ不足

大容量データの場合:

```python
# データセット設定で mmap を有効化（自動）
quick_debug: false  # これでmmap_mode='r'が使われる
```

### 画像サイズの問題

Atari標準サイズ (210x160) から 64x64 へのリサイズは自動で行われます。
別のサイズを使いたい場合:

```yaml
data:
  atari_config:
    img_size: 84  # DQN標準サイズ
```

## 今後の拡張

- [ ] オンライン生成モードの完成 (AtariOnlineDataset)
- [ ] 訓練済みポリシー（DQNなど）からのデータ生成
- [ ] プランニング評価の実装
- [ ] Pixel Mapperの実装（座標変換用）
- [ ] フレームスタックの完全サポート

## 参考文献

- [Gymnasium Atari Environments](https://gymnasium.farama.org/environments/atari/)
- [The Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
- [DQN Paper](https://www.nature.com/articles/nature14236) - Human-level control through deep reinforcement learning
