# Atari Pacman Integration Summary

このドキュメントは、PLDMリポジトリへのAtari Pacman統合の完全な概要を提供します。

## 🎯 完了した作業

### 1. コア実装ファイル

#### データ構造とコンフィグ
- **`enums.py`**: `AtariSample` と `AtariDatasetConfig` の定義
  - デフォルト環境: `ALE/Pacman-v5` (5 actions)
  - 画像処理、フレームスタック、正規化などの設定

#### データセット実装
- **`data/atari_dataset.py`**: `AtariDataset` クラス
  - NPZ/PKLファイルからのデータロード
  - 画像の前処理（リサイズ、正規化、グレースケール変換）
  - スライディングウィンドウサンプリング
  - メモリマッピングサポート（大規模データセット対応）

#### Factory統合
- **`pldm/data/enums.py`**: `DatasetType.Atari` の追加
- **`pldm/data/dataset_factory.py`**: `_create_atari_datasets()` メソッドの追加

### 2. ツールとスクリプト

#### データ生成
- **`data_generation/generate_data.py`**:
  - Gymnasiumを使ったAtari環境からのデータ収集
  - ランダムポリシーによるエピソード生成
  - エピソードのパディング機能
  - デフォルト環境: `ALE/Pacman-v5`

#### テスト
- **`test_dataset.py`**:
  - ダミーデータでの動作確認
  - 実データの検証
  - 画像の可視化

### 3. 設定ファイル

#### Pacman設定（推奨）
- **`configs/pacman.yaml`**:
  - 環境: `ALE/Pacman-v5`
  - アクション数: 5
  - ResNet18 バックボーン
  - VICReg + IDM 目的関数

#### MsPacman設定
- **`configs/mspacman.yaml`**:
  - 環境: `ALE/MsPacman-v5`
  - アクション数: 9
  - より複雑なゲームプレイ

### 4. ドキュメント

#### ユーザー向けガイド
- **`QUICK_START.md`**: 5ステップクイックスタートガイド
  - Pacman vs MsPacman の比較
  - トラブルシューティング
  - 他のゲームへの適用方法

- **`ATARI_SETUP_GUIDE.md`**: 詳細なセットアップガイド
  - インストール手順
  - データ生成の詳細
  - ハイパーパラメータ調整
  - 次のステップ

- **`README.md`**: 技術的な詳細
  - 実装の詳細
  - データフォーマット
  - ディレクトリ構造

## 📂 ディレクトリ構造

```
pldm_envs/atari/
├── __init__.py
├── enums.py                       # データ構造と設定
├── README.md                      # 技術ドキュメント
├── QUICK_START.md                 # クイックスタート
├── INTEGRATION_SUMMARY.md         # このファイル
│
├── data/
│   ├── __init__.py
│   └── atari_dataset.py          # データセットクラス
│
├── data_generation/
│   └── generate_data.py          # データ生成スクリプト
│
├── configs/
│   ├── pacman.yaml               # Pacman設定（推奨）
│   └── mspacman.yaml             # MsPacman設定
│
└── test_dataset.py               # テストスクリプト
```

## 🔑 重要な設計判断

### 1. Pacman vs MsPacman

| 特徴 | Pacman | MsPacman |
|------|--------|----------|
| **アクション数** | 5 | 9 |
| **ゲームプレイ** | シンプル | 複雑 |
| **ゴーストの動き** | パターン化 | 半ランダム |
| **推奨用途** | 初期実験、デバッグ | 本格的な訓練 |

**デフォルトをPacmanにした理由**:
- アクション空間が小さく、学習しやすい
- デバッグと初期実験に適している
- ユーザーの明示的な要望

### 2. ライブラリ選択

**Gymnasium (not Gym)**:
- Gym は deprecated（2022年以降）
- Gymnasium はアクティブにメンテナンス中
- ALE v0.9+ では ROMs が自動的に含まれる
- 環境名には `ALE/` プレフィックスが必要 (`ALE/Pacman-v5`)

### 3. データフォーマット

**NPZ形式を採用**:
```python
{
    'observations': [N_episodes, T, H, W, C],  # uint8
    'actions': [N_episodes, T-1],              # int
    'rewards': [N_episodes, T-1],              # float32
    'dones': [N_episodes, T-1],                # bool
}
```

**パディング戦略**:
- 固定長パディング（推奨）: `--pad_length 1000`
- 可変長も対応（実装は複雑）

### 4. 画像処理パイプライン

1. **リサイズ**: 210×160 → 64×64
2. **正規化**: 0-255 → 0-1
3. **チャネル順序**: [H, W, C] → [C, H, W]
4. **フレームスタック**: オプション（DQNスタイル）
5. **グレースケール**: オプション

## 🚀 使用方法

### クイックスタート（5ステップ）

```bash
# 1. インストール
pip install 'gymnasium[atari]' opencv-python

# 2. データ生成（デバッグ用）
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name ALE/Pacman-v5 \
  --n_episodes 10 \
  --output_path data/atari/pacman_debug.npz \
  --pad_length 100

# 3. テスト
python pldm_envs/atari/test_dataset.py \
  --data_path data/atari/pacman_debug.npz

# 4. 設定ファイル編集
# configs/pacman.yaml の data_path を更新

# 5. 訓練開始
python pldm/train.py \
  --configs pldm_envs/atari/configs/pacman.yaml \
  --values quick_debug=true epochs=1
```

### 本格的な訓練

```bash
# 1000エピソード生成（3-6時間）
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name ALE/Pacman-v5 \
  --n_episodes 1000 \
  --output_path data/atari/pacman_train.npz \
  --pad_length 1000 \
  --seed 42

# 訓練実行
python pldm/train.py --configs pldm_envs/atari/configs/pacman.yaml
```

## 🔧 主要な設定パラメータ

### データセット設定

```yaml
data:
  dataset_type: Atari
  atari_config:
    env_name: "ALE/Pacman-v5"
    data_path: data/atari/pacman_train.npz
    batch_size: 32
    sample_length: 17
    img_size: 64
    grayscale: false
    frame_stack: 1
    normalize_images: true
```

### モデル設定

```yaml
hjepa:
  level1:
    backbone:
      arch: resnet18  # Atariに適している
    action_dim: 5     # Pacman: 5, MsPacman: 9
```

### 目的関数

```yaml
objectives_l1:
  objectives:
    - VICReg  # 表現学習
    - IDM     # アクション予測

  vicreg:
    lambda_inv: 25.0
    lambda_var: 25.0
    lambda_cov: 1.0

  idm:
    action_dim: 5
```

## 🎮 他のAtariゲームへの適用

### サポートされているゲーム

| ゲーム | 環境名 | アクション数 |
|--------|--------|------------|
| **Pacman** | `ALE/Pacman-v5` | 5 |
| **MsPacman** | `ALE/MsPacman-v5` | 9 |
| **Pong** | `ALE/Pong-v5` | 6 |
| **Breakout** | `ALE/Breakout-v5` | 4 |
| **SpaceInvaders** | `ALE/SpaceInvaders-v5` | 6 |

詳細: [Gymnasium Atari Environments](https://gymnasium.farama.org/environments/atari/)

### 新しいゲームを追加する手順

1. **データ生成**:
```bash
python pldm_envs/atari/data_generation/generate_data.py \
  --env_name ALE/Pong-v5 \
  --n_episodes 1000 \
  --output_path data/atari/pong_train.npz \
  --pad_length 1000
```

2. **設定ファイルをコピー**:
```bash
cp pldm_envs/atari/configs/pacman.yaml \
   pldm_envs/atari/configs/pong.yaml
```

3. **設定を編集**:
```yaml
data:
  atari_config:
    env_name: "ALE/Pong-v5"
    data_path: data/atari/pong_train.npz

hjepa:
  level1:
    action_dim: 6  # Pongのアクション数

objectives_l1:
  idm:
    action_dim: 6
```

4. **訓練実行**:
```bash
python pldm/train.py --configs pldm_envs/atari/configs/pong.yaml
```

## ⚠️ 既知の制限と今後の拡張

### 現在の制限

1. **オンライン生成モード**: 実装済みだが未完成
2. **ポリシー**: ランダムポリシーのみ対応
3. **評価**: プランニング評価は未実装
4. **並列化**: データ生成の並列化は未対応

### 今後の拡張予定

- [ ] 訓練済みポリシー（DQNなど）からのデータ生成
- [ ] オンライン生成モードの完成
- [ ] プランニング評価の実装
- [ ] Pixel Mapperの実装（座標変換用）
- [ ] データ生成の並列化

## 🐛 トラブルシューティング

### よくあるエラー

#### 1. `Environment doesn't exist`
```
gymnasium.error.NameNotFound: Environment `MsPacman` doesn't exist.
```

**解決**: `ALE/` プレフィックスを付ける
```bash
# ❌ 間違い
--env_name MsPacman-v5

# ✅ 正しい
--env_name ALE/MsPacman-v5
```

#### 2. `ImportError: No module named 'gymnasium'`
```bash
pip install 'gymnasium[atari]'
```

#### 3. `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

#### 4. メモリ不足
```yaml
data:
  atari_config:
    batch_size: 16
    crop_length: 1000
  num_workers: 0
```

#### 5. アクション数のミスマッチ
設定ファイルの `action_dim` を環境に合わせて変更:
```yaml
hjepa:
  level1:
    action_dim: 5  # Pacman
```

## 📚 参考資料

### 公式ドキュメント
- [Gymnasium Atari Environments](https://gymnasium.farama.org/environments/atari/)
- [ALE-py PyPI](https://pypi.org/project/ale-py/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

### 論文
- [DQN: Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [OCAtari: Object-Centric Atari 2600 Reinforcement Learning Environments](https://arxiv.org/pdf/2306.08649)

### プロジェクトドキュメント
- [QUICK_START.md](QUICK_START.md) - 5ステップクイックスタート
- [ATARI_SETUP_GUIDE.md](../../../ATARI_SETUP_GUIDE.md) - 詳細セットアップ
- [README.md](README.md) - 技術詳細
- [adding_new_environment_guide.md](../../../adding_new_environment_guide.md) - 新環境追加ガイド

## ✅ 統合チェックリスト

完了した項目:

- [x] `AtariSample` と `AtariDatasetConfig` の定義
- [x] `AtariDataset` クラスの実装
- [x] `DatasetFactory` への統合
- [x] データ生成スクリプト (`generate_data.py`)
- [x] テストスクリプト (`test_dataset.py`)
- [x] Pacman設定ファイル (`configs/pacman.yaml`)
- [x] MsPacman設定ファイル (`configs/mspacman.yaml`)
- [x] クイックスタートガイド (`QUICK_START.md`)
- [x] 詳細セットアップガイド (`ATARI_SETUP_GUIDE.md`)
- [x] 技術ドキュメント (`README.md`)
- [x] デフォルト環境をPacmanに変更
- [x] Gymnasium v5 互換性（`ALE/` プレフィックス）
- [x] メモリマッピングサポート
- [x] 画像前処理パイプライン
- [x] トラブルシューティングガイド

## 🎉 結論

Atari Pacman統合は完全に完了しました。ユーザーは以下を実行できます:

1. **すぐに開始**: `QUICK_START.md` の5ステップで動作確認
2. **本格的な訓練**: 1000+エピソードのデータで訓練
3. **他のゲームに拡張**: Pong, Breakout, SpaceInvadersなど
4. **カスタマイズ**: 設定ファイルで細かく調整可能

すべてのファイルは適切に作成され、ドキュメント化され、テスト可能な状態になっています。

---

**作成日**: 2025-12-30
**バージョン**: 1.0
**ステータス**: ✅ 完了
