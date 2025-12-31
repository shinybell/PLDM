# Long-Horizon MiniGrid Environments

段階的に複雑さを増した3つのカスタムMiniGrid環境です。ロングホライズン実験のために設計されています。

## 環境概要

### Level 1: Medium Complexity (MiniGrid-LongHorizon-Level1-v0)

- **グリッドサイズ**: 16x16
- **壁パターン**: L字型の壁で2-3個の部屋を作成
- **予想ホライズン**: ランダムポリシーで約300-400ステップ
- **max_steps**: 512
- **難易度**: 中

シンプルな8x8 Empty環境とより複雑な迷路環境の中間に位置する環境です。

### Level 2: High Complexity (MiniGrid-LongHorizon-Level2-v0)

- **グリッドサイズ**: 24x24
- **壁パターン**: 複数の部屋、廊下、行き止まりを含む複雑な構造
- **予想ホライズン**: ランダムポリシーで約600-800ステップ
- **max_steps**: 1024
- **難易度**: 高

複数の相互接続された部屋があり、慎重なナビゲーションが必要です。

### Level 3: Very High Complexity (MiniGrid-LongHorizon-Level3-v0)

- **グリッドサイズ**: 32x32
- **壁パターン**: 迷路のような構造で、多くの廊下と障害物
- **予想ホライズン**: ランダムポリシーで約1000-1500ステップ
- **max_steps**: 2048
- **難易度**: 非常に高

最も挑戦的なナビゲーションタスクで、広範な探索が必要です。

## インストール

```bash
pip install minigrid gymnasium
```

## 使い方

### 1. 環境のテスト

環境が正しく動作するか確認:

```bash
# Level 1をテスト（デバッグモード）
python pldm_envs/minigrid/test_minigrid.py --env_name MiniGrid-LongHorizon-Level1-v0 --n_episodes 5

# Level 2をテスト
python pldm_envs/minigrid/test_minigrid.py --env_name MiniGrid-LongHorizon-Level2-v0 --n_episodes 5

# Level 3をテスト
python pldm_envs/minigrid/test_minigrid.py --env_name MiniGrid-LongHorizon-Level3-v0 --n_episodes 5
```

### 2. データ生成

各レベルのトレーニング用と検証用のデータを生成します。

#### Level 1 (16x16)

```bash
# トレーニング用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level1-v0 \
    --n_episodes 20000 \
    --max_steps 512 \
    --resize 64 \
    --seed 42 \
    --output_path data/minigrid/long_horizon_level1_train.npz

# 検証用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level1-v0 \
    --n_episodes 10000 \
    --max_steps 512 \
    --resize 64 \
    --seed 100 \
    --output_path data/minigrid/long_horizon_level1_val.npz
```

#### Level 2 (24x24)

```bash
# トレーニング用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level2-v0 \
    --n_episodes 20000 \
    --max_steps 1024 \
    --resize 64 \
    --seed 42 \
    --output_path data/minigrid/long_horizon_level2_train.npz

# 検証用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level2-v0 \
    --n_episodes 10000 \
    --max_steps 1024 \
    --resize 64 \
    --seed 100 \
    --output_path data/minigrid/long_horizon_level2_val.npz
```

#### Level 3 (32x32)

```bash
# トレーニング用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level3-v0 \
    --n_episodes 20000 \
    --max_steps 2048 \
    --resize 64 \
    --seed 42 \
    --output_path data/minigrid/long_horizon_level3_train.npz

# 検証用データ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level3-v0 \
    --n_episodes 10000 \
    --max_steps 2048 \
    --resize 64 \
    --seed 100 \
    --output_path data/minigrid/long_horizon_level3_val.npz
```

#### デバッグ用（少量データ）

```bash
# Level 1 デバッグ
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-LongHorizon-Level1-v0 \
    --n_episodes 10 \
    --max_steps 512 \
    --resize 64 \
    --output_path data/minigrid/long_horizon_level1_debug.npz
```

### 3. エピソードの可視化

生成したデータを確認:

```bash
# Level 1の最初のエピソードを可視化
python pldm_envs/minigrid/visualize_episode.py \
    --data_path data/minigrid/long_horizon_level1_debug.npz \
    --episode_idx 0 \
    --output_dir visualizations/minigrid/level1

# Level 2の可視化
python pldm_envs/minigrid/visualize_episode.py \
    --data_path data/minigrid/long_horizon_level2_debug.npz \
    --episode_idx 0 \
    --output_dir visualizations/minigrid/level2

# Level 3の可視化
python pldm_envs/minigrid/visualize_episode.py \
    --data_path data/minigrid/long_horizon_level3_debug.npz \
    --episode_idx 0 \
    --output_dir visualizations/minigrid/level3
```

### 4. モデル学習

```bash
# Level 1で学習
python pldm/train.py --config pldm_envs/minigrid/configs/long_horizon_level1.yaml

# Level 2で学習
python pldm/train.py --config pldm_envs/minigrid/configs/long_horizon_level2.yaml

# Level 3で学習
python pldm/train.py --config pldm_envs/minigrid/configs/long_horizon_level3.yaml
```

## ディレクトリ構造

```
pldm_envs/minigrid/
├── LONG_HORIZON.md                    # このファイル
├── README.md                          # 基本的な使い方
├── envs/
│   ├── __init__.py
│   └── long_horizon_envs.py          # カスタム環境の実装
├── configs/
│   ├── long_horizon_level1.yaml      # Level 1設定
│   ├── long_horizon_level2.yaml      # Level 2設定
│   └── long_horizon_level3.yaml      # Level 3設定
└── data_generation/
    └── generate_data.py              # データ生成スクリプト
```

## 環境の設計思想

### 段階的な複雑さの増加

1. **Level 1 (16x16)**: 基本的な壁構造で、複数の経路がある程度明確
2. **Level 2 (24x24)**: 複数の部屋と廊下、行き止まりを追加
3. **Level 3 (32x32)**: 密度の高い迷路構造で、最適経路が非常に不明確

### ロングホライズンの達成方法

- **グリッドサイズの増加**: より広い空間を探索する必要がある
- **壁の複雑化**: 直線的な移動が困難になり、回り道が必要
- **行き止まりの追加**: 探索と後戻りが必要
- **最適経路の不明確化**: ランダムポリシーでより長いエピソードが発生

## 研究計画との対応

### Phase 1: ベースライン確立

1. Level 1でデータ生成・学習
2. Level 2でデータ生成・学習
3. Level 3でデータ生成・学習
4. 各レベルでのベースラインPLDMの性能評価

### Phase 2: ホライズン別の精度比較

1. 各レベルでの予測精度の測定
2. ホライズンが長くなるにつれての精度劣化の分析
3. 環境の複雑さと予測精度の関係を調査

### Phase 3: 離散化手法の効果検証

1. VQ-VAE/FSQを各レベルで実装
2. 各レベルでの精度向上効果を測定
3. 複雑な環境ほど離散化の効果が大きいかを検証

## 技術的詳細

### 環境の実装

- MiniGridの`MiniGridEnv`クラスを継承
- `_gen_grid()`メソッドで壁配置を定義
- エージェントとゴールの配置はランダム化
- 再現性のためnumpy seedを固定可能

### データフォーマット

生成されるデータは通常のMiniGridデータと同じフォーマット:

- **observations**: `[N_episodes, T, H, W, C]` - RGB画像観測
- **actions**: `[N_episodes, T-1]` - 離散アクション（0-6）
- **rewards**: `[N_episodes, T-1]` - 報酬
- **dones**: `[N_episodes, T-1]` - 終端フラグ

### モデル設定の違い

- **Level 1**: ResNet18（軽量）
- **Level 2-3**: ResNet34（より大きな受容野が必要）
- エポック数: Level 1 (150) → Level 2 (200) → Level 3 (250)

## 予想される結果

### ホライズン別の予測精度

| Level | Grid Size | Expected Horizon | Prediction Difficulty |
|-------|-----------|------------------|----------------------|
| 1     | 16x16     | 300-400 steps    | Medium               |
| 2     | 24x24     | 600-800 steps    | High                 |
| 3     | 32x32     | 1000-1500 steps  | Very High            |

### 評価指標

1. **短期予測精度** (1-5ステップ先): すべてのレベルで高精度
2. **中期予測精度** (10-20ステップ先): 徐々に劣化
3. **長期予測精度** (50+ステップ先): レベル3で大きく劣化することを予想

## トラブルシューティング

### 環境が登録されない

```python
# pldm_envs/minigrid/__init__.py をインポートしてから使用
import pldm_envs.minigrid
import gymnasium as gym

env = gym.make("MiniGrid-LongHorizon-Level1-v0")
```

### データ生成が遅い

- Level 3は特に生成に時間がかかります（1エピソード最大2048ステップ）
- `--n_episodes` を減らしてデバッグすることを推奨
- `opencv-python` のインストールで画像処理を高速化

### メモリ不足

- Level 3のデータは特に大きくなります
- バッチサイズやエピソード数を調整
- `crop_length` パラメータを設定してエピソードを短縮

## 参考

- [MiniGrid Documentation](https://minigrid.farama.org/)
- [MiniGrid Environment Creation Tutorial](https://minigrid.farama.org/content/create_env_tutorial/)
- [Gymnasium](https://gymnasium.farama.org/)
