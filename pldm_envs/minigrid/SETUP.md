# MiniGrid環境のセットアップガイド

新しい環境でMiniGridを使用するための手順です。

## 前提条件

- Python 3.8以上
- pip

## インストール手順

### 1. リポジトリのクローン

```bash
git clone <repository_url>
cd PLDM
```

### 2. 必要なパッケージのインストール

```bash
# 基本的なPyTorchパッケージ（既にインストール済みの場合はスキップ）
pip install torch torchvision

# MiniGrid環境のインストール
pip install minigrid

# その他の依存関係
pip install numpy tqdm pillow
```

### 3. 環境変数の設定

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

または、実行時に毎回指定：

```bash
PYTHONPATH=$PWD:$PYTHONPATH python your_script.py
```

## データ生成

### テスト用データ（10エピソード）

```bash
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 10 \
  --obs_size 72 \
  --output_path data/minigrid/level1_72x72_train.npz
```

### 訓練用データ（1000エピソード）

```bash
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 1000 \
  --obs_size 72 \
  --output_path data/minigrid/level1_72x72_train.npz
```

## トラブルシューティング

### エラー: `ModuleNotFoundError: No module named 'minigrid'`

**原因**: MiniGrid環境がインストールされていない

**解決策**:
```bash
pip install minigrid
```

### エラー: `ModuleNotFoundError: No module named 'pldm_envs'`

**原因**: PYTHONPATHが設定されていない

**解決策**:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
# または
cd /path/to/PLDM
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### エラー: `ModuleNotFoundError: No module named 'gymnasium'`

**原因**: Gymnasiumがインストールされていない（MiniGridの依存関係）

**解決策**:
```bash
pip install gymnasium
```

## 利用可能な環境

- `MiniGrid-LongHorizon-Level1-v0`: 簡単（8x8グリッド）
- `MiniGrid-LongHorizon-Level2-v0`: 中級（16x16グリッド）
- `MiniGrid-LongHorizon-Level3-v0`: 難しい（24x24グリッド）

## データ生成オプション

| オプション | デフォルト | 説明 |
|----------|-----------|------|
| `--env_name` | MiniGrid-LongHorizon-Level1-v0 | 環境名 |
| `--n_episodes` | 1000 | エピソード数 |
| `--obs_size` | 72 | 観測画像サイズ（64, 72, 84など） |
| `--output_path` | (必須) | 出力ファイルパス (.npz) |
| `--channel_first` | False | CHW形式で保存 |
| `--normalize` | False | 画像を[0,1]に正規化 |
| `--max_steps` | None | 最大ステップ数 |

## 例

### Level 2の大規模データセット生成

```bash
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level2-v0 \
  --n_episodes 5000 \
  --obs_size 72 \
  --output_path data/minigrid/level2_72x72_large.npz
```

### 64x64画像での生成

```bash
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 1000 \
  --obs_size 64 \
  --output_path data/minigrid/level1_64x64_train.npz
```

## 次のステップ

データ生成が完了したら：
1. [PLDM_INTEGRATION.md](PLDM_INTEGRATION.md)を参照してPLDM学習を実行
2. [QUICKSTART_LONG_HORIZON.md](QUICKSTART_LONG_HORIZON.md)でMiniGrid環境の詳細を確認

## 参考

- MiniGrid公式ドキュメント: https://minigrid.farama.org/
- PLDM統合ガイド: [PLDM_INTEGRATION.md](PLDM_INTEGRATION.md)
