# Long-Horizon Environments - Quick Start Guide

3段階の複雑さを持つカスタムMiniGrid環境のクイックスタートガイドです。

## 環境の特徴

**すべての環境で統一:**
- Grid Size: 24x24 (固定)
- Max Steps: 256 (固定)
- 違いは壁の複雑さのみ

| Level | Wall Complexity | Description |
|-------|----------------|-------------|
| 1     | Simple         | L字型の壁で2-3個の部屋 |
| 2     | Medium         | 複数の部屋と廊下 |
| 3     | Complex        | 密度の高い迷路構造 |

## 🚀 簡単スタート（推奨）

すべてのレベルのテストデータ生成と可視化を自動化するスクリプトを用意しています。

```bash
# すべてのレベルを実行（各レベル10エピソード、約5分）
bash pldm_envs/minigrid/test_all_levels.sh

# 特定のレベルのみ実行
bash pldm_envs/minigrid/test_all_levels.sh --only-level 1

# エピソード数を指定
bash pldm_envs/minigrid/test_all_levels.sh --n-episodes 5

# 既存データを使って可視化のみ
bash pldm_envs/minigrid/test_all_levels.sh --visualize-only
```

**注意**: すべてのレベルで同じグリッドサイズ(24x24)とmax_steps(256)を使用しているため、データ生成時間はすべて同じです。

スクリプトは以下を自動的に実行します：
- ✅ 各レベルのテストデータ生成
- ✅ 各レベルの最初の2エピソードを可視化
- ✅ 結果をディレクトリごとに整理

## 環境の設計思想

### 統一された設定での複雑さの比較

すべての環境で以下を統一:
- **Grid Size: 24x24** - 同じ探索空間
- **Max Steps: 256** - 同じホライズン

### 壁の複雑さによる違い

1. **Level 1 (Simple)**:
   - L字型の壁で基本的な部屋構造
   - 経路が比較的明確
   - ナビゲーションが容易

2. **Level 2 (Medium)**:
   - 複数の部屋と廊下を持つグリッド構造
   - 行き止まりと迂回路が存在
   - 中程度のナビゲーション難易度

3. **Level 3 (Complex)**:
   - 密度の高い迷路構造
   - 多数の行き止まりと複雑な経路
   - ナビゲーションが最も困難

**この設計により、グリッドサイズやホライズンの違いを排除し、純粋に壁の複雑さが予測精度に与える影響を測定できます。**

## トレーニングデータを生成（オプション）

スクリプトではなく手動でデータ生成する場合のコマンドです。

**注意**: すべてのレベルで同じmax_steps(256)を使用

### Level 1 (Simple Walls)

```bash
# トレーニング用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level1-v0 \
    --n_episodes 20000 \
    --max_steps 256 \
    --resize 64 \
    --seed 42 \
    --output_path data/minigrid/long_horizon_level1_train.npz

# 検証用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level1-v0 \
    --n_episodes 10000 \
    --max_steps 256 \
    --resize 64 \
    --seed 100 \
    --output_path data/minigrid/long_horizon_level1_val.npz
```

### Level 2 (Medium Walls)

```bash
# トレーニング用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level2-v0 \
    --n_episodes 20000 \
    --max_steps 256 \
    --resize 64 \
    --seed 42 \
    --output_path data/minigrid/long_horizon_level2_train.npz

# 検証用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level2-v0 \
    --n_episodes 10000 \
    --max_steps 256 \
    --resize 64 \
    --seed 100 \
    --output_path data/minigrid/long_horizon_level2_val.npz
```

### Level 3 (Complex Walls)

```bash
# トレーニング用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level3-v0 \
    --n_episodes 20000 \
    --max_steps 256 \
    --resize 64 \
    --seed 42 \
    --output_path data/minigrid/long_horizon_level3_train.npz

# 検証用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level3-v0 \
    --n_episodes 10000 \
    --max_steps 256 \
    --resize 64 \
    --seed 100 \
    --output_path data/minigrid/long_horizon_level3_val.npz
```

## モデルを学習

```bash
# Level 1で学習
python pldm/train.py --config pldm_envs/minigrid/configs/long_horizon_level1.yaml

# Level 2で学習
python pldm/train.py --config pldm_envs/minigrid/configs/long_horizon_level2.yaml

# Level 3で学習
python pldm/train.py --config pldm_envs/minigrid/configs/long_horizon_level3.yaml
```

**注意**: 設定ファイル内の `data_path` と `val_path` を実際のデータパスに更新してください。

## 研究での使用

これらの環境は以下の研究質問に答えるために設計されています：

1. **壁の複雑さと予測精度**: 壁が複雑になると予測精度はどう変化するか？
2. **環境の構造の影響**: ナビゲーションの難しさが長期予測にどう影響するか？
3. **離散化の効果**: VQ-VAE/FSQは複雑な環境でより効果的か？

グリッドサイズとホライズンを統一することで、これらの質問に体系的に答えることができます。

## 次のステップ

詳細な情報は以下を参照:
- [README_TEST_SCRIPT.md](README_TEST_SCRIPT.md) - テストスクリプトの詳細な使い方
- [LONG_HORIZON.md](LONG_HORIZON.md) - 完全なドキュメント
- [README.md](README.md) - 基本的なMiniGrid環境の使い方
