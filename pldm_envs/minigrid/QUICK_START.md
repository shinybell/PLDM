# MiniGrid Quick Start Guide

MiniGridを使ってPLDMのロングホライズン実験を開始するためのクイックスタートガイドです。

## 📋 前提条件

```bash
# MiniGridのインストール
pip install minigrid

# オプション（画像処理の高速化）
pip install opencv-python
```

## 🚀 クイックスタート（3ステップ）

### Step 1: 環境のテスト

まず、MiniGrid環境が正しく動作するか確認します：

```bash
python pldm_envs/minigrid/test_minigrid.py --test_env
```

✅ 成功すると以下のように表示されます：
```
✓ Environment created successfully
✓ Environment test passed!
```

### Step 2: デバッグ用データの生成

小規模なデータセットで動作確認します（約1分）：

```bash
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 10 \
    --max_steps 200 \
    --resize 64 \
    --output_path data/minigrid/empty_8x8_debug.npz
```

### Step 3: データセットのテスト

生成したデータが正しく読み込めるか確認します：

```bash
python pldm_envs/minigrid/test_minigrid.py --test_dataset \
    --data_path data/minigrid/empty_8x8_debug.npz
```

✅ 全て成功すれば、本格的なデータ生成に進めます！

## 📊 本番用データの生成

### ホライズン200（推奨：まずはこちらから）

```bash
# 訓練用データ（20,000エピソード、約30-60分）
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 20000 \
    --max_steps 200 \
    --resize 64 \
    --seed 42 \
    --output_path data/minigrid/empty_8x8_h200_train.npz

# 検証用データ（10,000エピソード、約15-30分）
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 10000 \
    --max_steps 200 \
    --resize 64 \
    --seed 100 \
    --output_path data/minigrid/empty_8x8_h200_val.npz
```

### ホライズン256（デフォルト設定）

```bash
# 訓練用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 20000 \
    --max_steps 256 \
    --resize 64 \
    --seed 42 \
    --output_path data/minigrid/empty_8x8_h256_train.npz

# 検証用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 10000 \
    --max_steps 256 \
    --resize 64 \
    --seed 100 \
    --output_path data/minigrid/empty_8x8_h256_val.npz
```

## ⚙️ 設定ファイルの更新

生成したデータのパスを設定ファイルに記入します：

```bash
# configs/empty_8x8.yaml を編集
vim pldm_envs/minigrid/configs/empty_8x8.yaml
```

以下の部分を更新：

```yaml
data:
  minigrid_config:
    data_path: data/minigrid/empty_8x8_h200_train.npz
    val_path: data/minigrid/empty_8x8_h200_val.npz
    max_steps: 200  # ホライズン
```

## 🎯 次のステップ

1. **ベースラインモデルの学習**
   ```bash
   # TODO: PLDMの学習スクリプトが完成したら追加
   python train.py --config pldm_envs/minigrid/configs/empty_8x8.yaml
   ```

2. **VQ-VAEの実装**
   - 潜在空間の離散化モジュールを追加

3. **FSQの実装**
   - Finite Scalar Quantizationを実装

4. **評価と比較**
   - ベースライン vs VQ vs FSQの性能比較

## 💡 Tips

### データ生成を高速化したい

```bash
# 並列で複数のデータ生成を実行
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 5000 \
    --max_steps 200 \
    --resize 64 \
    --seed 42 \
    --output_path data/minigrid/empty_8x8_part1.npz &

python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 5000 \
    --max_steps 200 \
    --resize 64 \
    --seed 142 \
    --output_path data/minigrid/empty_8x8_part2.npz &

# 完了後、結合（別途スクリプトが必要）
```

### メモリを節約したい

```yaml
# 設定ファイルで以下を設定
data:
  minigrid_config:
    quick_debug: false  # メモリマップモード
    crop_length: 10000  # データセット長を制限
```

### レンダリングして確認したい

```bash
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 5 \
    --max_steps 200 \
    --render \
    --output_path /tmp/test.npz
```

## ❓ トラブルシューティング

### `ModuleNotFoundError: No module named 'minigrid'`

```bash
pip install minigrid
```

### データ生成が遅い

```bash
# opencv-pythonをインストールして高速化
pip install opencv-python
```

### メモリエラーが出る

```bash
# エピソード数を減らす
--n_episodes 10000

# または、バッチサイズを小さくする（設定ファイル）
batch_size: 16
```

## 📚 参考資料

- [MiniGrid Documentation](https://minigrid.farama.org/)
- [Empty Environment](https://minigrid.farama.org/environments/minigrid/EmptyEnv/)
- [詳細なREADME](./README.md)
