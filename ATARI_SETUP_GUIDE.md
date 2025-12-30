# Atari (Pacman) セットアップガイド

このガイドでは、PLDMでAtari Pacmanを使って訓練を開始するまでの手順を説明します。

> **Note**: PacmanとMsPacmanの違い
> - **Pacman**: 5 actions（シンプル、ゴーストの動きがパターン化）
> - **MsPacman**: 9 actions（より複雑、ゴーストの動きが半ランダム）
>
> このガイドではPacmanを使用しますが、MsPacmanも同様の手順で使用可能です。

## クイックスタート

### 1. 依存関係のインストール

```bash
# Gymnasium + Atari環境
pip install 'gymnasium[atari]'

# 画像処理（オプション、推奨）
pip install opencv-python
```

**参考情報**:
- [Gymnasium Documentation](https://gymnasium.farama.org/environments/atari/)
- [ALE-py PyPI](https://pypi.org/project/ale-py/)

### 2. データの生成

```bash
# デバッグ用データ（10エピソード、数分で完了）
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name ALE/Pacman-v5 \
  --n_episodes 10 \
  --output_path data/atari/pacman_debug.npz \
  --pad_length 100

# 訓練用データ（1000エピソード、時間がかかる）
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name ALE/Pacman-v5 \
  --n_episodes 1000 \
  --output_path data/atari/pacman_train.npz \
  --pad_length 1000 \
  --seed 42
```

**データ生成時間の目安**:
- 10エピソード: 約5-10分
- 100エピソード: 約30-60分
- 1000エピソード: 約3-6時間

### 3. データセットのテスト

```bash
# ダミーデータで動作確認
python pldm_envs/atari/test_dataset.py

# 生成したデータで動作確認
python pldm_envs/atari/test_dataset.py \
  --data_path data/atari/pacman_debug.npz
```

成功すると、以下が表示されます:
- データセット長
- サンプルの形状（states, actions）
- 画像の可視化（`test_atari_visualization.png`）

### 4. 設定ファイルの編集

[pldm_envs/atari/configs/pacman.yaml](pldm_envs/atari/configs/pacman.yaml) を編集:

```yaml
data:
  atari_config:
    # 生成したデータのパスに変更
    data_path: data/atari/pacman_debug.npz  # または pacman_train.npz
    val_path: null  # 検証データがあれば指定
```

### 5. 訓練の実行

```bash
# クイックデバッグモード（動作確認用）
python pldm/train.py \
  --configs pldm_envs/atari/configs/pacman.yaml \
  --values \
    quick_debug=true \
    epochs=1 \
    data.atari_config.crop_length=50

# 本格的な訓練
python pldm/train.py \
  --configs pldm_envs/atari/configs/pacman.yaml
```

---

## 詳細説明

### データ生成のオプション

```bash
python pldm_envs/atari/data_generation/generate_data.py --help

主要なオプション:
  --env_name: Atari環境名 (デフォルト: ALE/Pacman-v5)
  --n_episodes: エピソード数
  --output_path: 出力ファイルパス (.npz)
  --pad_length: エピソード長のパディング（推奨: 1000）
  --obs_type: 観測タイプ (rgb or grayscale)
  --frameskip: フレームスキップ (デフォルト: 4)
  --seed: ランダムシード
```

**パディングについて**:
- `--pad_length` を指定すると、すべてのエピソードが同じ長さになります
- 指定しない場合は可変長になり、データローダーの実装が複雑になります
- 推奨値: 1000-2000（MsPacmanの平均的なエピソード長より長め）

### 設定ファイルの主要パラメータ

```yaml
data:
  atari_config:
    # データソース
    data_path: data/atari/mspacman_train.npz
    val_path: data/atari/mspacman_val.npz  # オプション

    # バッチ設定
    batch_size: 32
    sample_length: 17  # サンプルのタイムステップ数

    # 画像設定
    img_size: 64       # リサイズ後のサイズ
    grayscale: false   # RGB を使用
    frame_stack: 4     # DQNスタイルのフレームスタック

    # 前処理
    normalize_images: true  # 0-255 -> 0-1 に正規化

# モデル設定
hjepa:
  level1:
    backbone:
      arch: resnet18  # Atariには resnet18 が適している
    action_dim: 9     # MsPacman のアクション数

# 訓練設定
epochs: 100
base_lr: 0.0001      # Atariには小さめのLRが適している
optimizer_type: Adam
```

### 他のAtariゲームを使う場合

1. **環境名を変更**:
```yaml
data:
  atari_config:
    env_name: "ALE/Pong-v5"  # または "ALE/Breakout-v5", "ALE/SpaceInvaders-v5" など
```

2. **アクション数を変更**:
```yaml
hjepa:
  level1:
    action_dim: 6  # Pong の場合
```

**主要なゲームのアクション数**:
- ALE/Pacman-v5: 5 actions（シンプル、推奨）
- ALE/MsPacman-v5: 9 actions（より複雑）
- ALE/Pong-v5: 6 actions
- ALE/Breakout-v5: 4 actions
- ALE/SpaceInvaders-v5: 6 actions

詳細は[Gymnasium Atari Documentation](https://gymnasium.farama.org/environments/atari/)を参照。

### メモリ使用量の削減

大容量データの場合:

```yaml
data:
  atari_config:
    quick_debug: false  # mmap モードを使用（自動）
  num_workers: 0        # メモリ使用量を抑える
```

または、データを分割:
```bash
# 小さいバッチで複数回訓練
python pldm/train.py \
  --configs pldm_envs/atari/configs/mspacman.yaml \
  --values \
    data.atari_config.crop_length=5000 \
    data.atari_config.batch_size=16
```

---

## トラブルシューティング

### Q1: `ImportError: No module named 'gymnasium'`

```bash
pip install 'gymnasium[atari]'
```

### Q2: `ModuleNotFoundError: No module named 'cv2'`

OpenCVがない場合でもPyTorchで動作しますが、インストールを推奨:
```bash
pip install opencv-python
```

### Q3: データ生成が遅い

- `--n_episodes` を減らす（デバッグ時は10-100程度）
- フレームスキップを増やす: `--frameskip 8`
- 並列化は現在未対応（今後の拡張予定）

### Q4: メモリ不足エラー

```yaml
# 設定ファイルで調整
data:
  atari_config:
    batch_size: 16      # バッチサイズを減らす
    crop_length: 1000   # データセット長を制限
  num_workers: 0        # ワーカー数を減らす
```

### Q5: アクション数のエラー

```
RuntimeError: action_dim mismatch
```

設定ファイルで `hjepa.level1.action_dim` を環境に合わせて変更:
```yaml
hjepa:
  level1:
    action_dim: 9  # MsPacman の場合
```

### Q6: 訓練が進まない / NaN loss

- 学習率を下げる: `base_lr: 0.00005`
- バッチサイズを調整: `batch_size: 64`
- 正規化を確認: `normalize_images: true`
- フレームスタックを無効化して試す: `frame_stack: 1`

---

## 次のステップ

### 1. データ量を増やす

```bash
# より多くのエピソードでデータ生成
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name ALE/Pacman-v5 \
  --n_episodes 5000 \
  --output_path data/atari/pacman_large.npz \
  --pad_length 1000
```

### 2. ハイパーパラメータのチューニング

```bash
# 学習率を変える
python pldm/train.py \
  --configs pldm_envs/atari/configs/pacman.yaml \
  --values base_lr=0.0002

# バッチサイズを変える
python pldm/train.py \
  --configs pldm_envs/atari/configs/pacman.yaml \
  --values data.atari_config.batch_size=64
```

### 3. 他のゲームを試す

```bash
# Pong でデータ生成
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name ALE/Pong-v5 \
  --n_episodes 1000 \
  --output_path data/atari/pong_train.npz

# 設定ファイルをコピーして編集
cp pldm_envs/atari/configs/pacman.yaml pldm_envs/atari/configs/pong.yaml
# env_name と action_dim を変更

# 訓練
python pldm/train.py --configs pldm_envs/atari/configs/pong.yaml
```

### 4. 訓練済みポリシーからデータ生成（今後対応予定）

現在はランダムポリシーのみ対応。DQNなどの訓練済みポリシーからのデータ生成は今後実装予定です。

---

## 参考資料

### ドキュメント
- **Atari環境**: [Gymnasium Atari Environments](https://gymnasium.farama.org/environments/atari/)
- **ALE**: [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
- **PLDM**: [プロジェクトREADME](README.md)

### 論文
- DQN: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- OCAtari: [Object-Centric Atari 2600 Reinforcement Learning Environments](https://arxiv.org/pdf/2306.08649)

### 実装ガイド
- [新しい環境の追加ガイド](adding_new_environment_guide.md)
- [データセット作成の詳細](dataset_creation_detailed.md)
- [Atari README](pldm_envs/atari/README.md)

---

## サポート

問題が発生した場合:

1. [トラブルシューティング](#トラブルシューティング)を確認
2. テストスクリプトで動作確認: `python pldm_envs/atari/test_dataset.py`
3. Issueを報告: [GitHub Issues](https://github.com/anthropics/claude-code/issues)

---

**Sources**:
- [Gymnasium Atari](https://gymnasium.farama.org/environments/atari/)
- [ale-py PyPI](https://pypi.org/project/ale-py/)
- [Gymnasium Release Notes](https://gymnasium.farama.org/gymnasium_release_notes/index.html)
