# Atari Pacman クイックスタート

## 🎮 Pacman vs MsPacman

| ゲーム | アクション数 | 特徴 |
|--------|------------|------|
| **Pacman** | 5 actions | シンプル、ゴーストの動きがパターン化、学習しやすい |
| **MsPacman** | 9 actions | 複雑、ゴーストの動きが半ランダム、より難しい |

**推奨**: まずは **Pacman** から始めることを推奨します。

## 📋 セットアップ手順（5ステップ）

### 1️⃣ インストール (1分)

```bash
pip install 'gymnasium[atari]'
pip install opencv-python  # オプションだが推奨
```

### 2️⃣ データ生成 (5-10分)

```bash
# Pacman のデバッグデータを生成
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name ALE/Pacman-v5 \
  --n_episodes 10 \
  --output_path data/atari/pacman_debug.npz \
  --pad_length 100
```

✅ 成功すると `data/atari/pacman_debug.npz` が作成されます

### 3️⃣ テスト (30秒)

```bash
python pldm_envs/atari/test_dataset.py \
  --data_path data/atari/pacman_debug.npz
```

✅ 成功すると `test_atari_visualization.png` が作成されます

### 4️⃣ 設定ファイル編集 (1分)

`pldm_envs/atari/configs/pacman.yaml` を開いて、データパスを更新:

```yaml
data:
  atari_config:
    data_path: data/atari/pacman_debug.npz
    val_path: null
```

### 5️⃣ 訓練開始！

```bash
# クイックデバッグ（動作確認）
python pldm/train.py \
  --configs pldm_envs/atari/configs/pacman.yaml \
  --values \
    quick_debug=true \
    epochs=1 \
    data.atari_config.crop_length=50
```

## 🚀 本格的な訓練

### データを増やす

```bash
# 1000エピソード生成（3-6時間かかる）
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name ALE/Pacman-v5 \
  --n_episodes 1000 \
  --output_path data/atari/pacman_train.npz \
  --pad_length 1000 \
  --seed 42
```

### 設定を更新

```yaml
data:
  atari_config:
    data_path: data/atari/pacman_train.npz
```

### 訓練実行

```bash
python pldm/train.py --configs pldm_envs/atari/configs/pacman.yaml
```

## 🎯 MsPacman を使う場合

### データ生成

```bash
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name ALE/MsPacman-v5 \
  --n_episodes 10 \
  --output_path data/atari/mspacman_debug.npz \
  --pad_length 100
```

### 設定ファイル

`pldm_envs/atari/configs/mspacman.yaml` を使用（既に用意済み）:

```yaml
data:
  atari_config:
    env_name: "ALE/MsPacman-v5"
    data_path: data/atari/mspacman_debug.npz

hjepa:
  level1:
    action_dim: 9  # MsPacmanは9アクション

objectives_l1:
  idm:
    action_dim: 9
```

### 訓練実行

```bash
python pldm/train.py --configs pldm_envs/atari/configs/mspacman.yaml
```

## 🔧 トラブルシューティング

### ❌ `Environment MsPacman doesn't exist`

**原因**: 環境名が間違っている
**解決**: `ALE/` プレフィックスを付ける

```bash
# ❌ 間違い
--env_name MsPacman-v5

# ✅ 正しい
--env_name ALE/MsPacman-v5
```

### ❌ `ImportError: No module named 'gymnasium'`

```bash
pip install 'gymnasium[atari]'
```

### ❌ `ModuleNotFoundError: No module named 'cv2'`

```bash
pip install opencv-python
```

### ❌ メモリ不足

設定ファイルで調整:

```yaml
data:
  atari_config:
    batch_size: 16      # デフォルト: 32
    crop_length: 1000   # データセット長を制限
  num_workers: 0
```

### ❌ 訓練が進まない / NaN loss

```bash
python pldm/train.py \
  --configs pldm_envs/atari/configs/pacman.yaml \
  --values \
    base_lr=0.00005 \
    data.atari_config.batch_size=64
```

## 📚 詳細ドキュメント

- **セットアップガイド**: [ATARI_SETUP_GUIDE.md](../../../ATARI_SETUP_GUIDE.md)
- **詳細README**: [README.md](README.md)
- **新環境追加**: [adding_new_environment_guide.md](../../../adding_new_environment_guide.md)

## 🎲 他のゲームを試す

```bash
# Pong
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name ALE/Pong-v5 \
  --n_episodes 10 \
  --output_path data/atari/pong_debug.npz

# Breakout
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name ALE/Breakout-v5 \
  --n_episodes 10 \
  --output_path data/atari/breakout_debug.npz
```

**重要**: 各ゲームのアクション数が異なるので、設定ファイルの `action_dim` を変更してください:

- Pacman: 5 actions
- MsPacman: 9 actions
- Pong: 6 actions
- Breakout: 4 actions

## 📊 データ生成時間の目安

| エピソード数 | 所要時間 | 用途 |
|------------|---------|------|
| 10 | 5-10分 | デバッグ |
| 100 | 30-60分 | 小規模実験 |
| 1000 | 3-6時間 | 本格訓練 |
| 5000 | 15-30時間 | 大規模訓練 |

**Sources**:
- [Gymnasium Atari Environments](https://gymnasium.farama.org/environments/atari/)
- [Pacman Environment](https://gymnasium.farama.org/v0.28.0/environments/atari/pacman/)
- [MsPacman Environment](https://gymnasium.farama.org/environments/atari/ms_pacman/)
- [ALE Documentation](https://ale.farama.org/)
