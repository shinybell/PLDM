# Test Script Usage Guide

[test_all_levels.sh](test_all_levels.sh) の使い方ガイド

## 概要

このスクリプトは、3つのレベルすべてについて以下を自動実行します：

1. テストデータの生成（各レベルごと）
2. 最初の2エピソードの可視化
3. 結果の整理されたディレクトリへの保存

## 基本的な使い方

### すべてのレベルを実行

```bash
bash pldm_envs/minigrid/test_all_levels.sh
```

これにより以下が実行されます：
- Level 1: 10エピソード生成 + 可視化（約5-7分）
- Level 2: 10エピソード生成 + 可視化（約10-12分）
- Level 3: 10エピソード生成 + 可視化（約20-25分）

**合計所要時間**: 約35-45分

### 特定のレベルのみ実行

```bash
# Level 1のみ
bash pldm_envs/minigrid/test_all_levels.sh --only-level 1

# Level 2のみ
bash pldm_envs/minigrid/test_all_levels.sh --only-level 2

# Level 3のみ
bash pldm_envs/minigrid/test_all_levels.sh --only-level 3
```

### エピソード数を変更

```bash
# 各レベル5エピソードで実行（高速テスト）
bash pldm_envs/minigrid/test_all_levels.sh --n-episodes 5

# Level 1のみ3エピソード
bash pldm_envs/minigrid/test_all_levels.sh --only-level 1 --n-episodes 3
```

### 可視化のみ実行

既にデータが生成されている場合、可視化のみを再実行できます：

```bash
bash pldm_envs/minigrid/test_all_levels.sh --visualize-only

# 特定のレベルのみ
bash pldm_envs/minigrid/test_all_levels.sh --visualize-only --only-level 2
```

## オプション一覧

| オプション | 説明 | デフォルト値 | 例 |
|-----------|------|-------------|-----|
| `--n-episodes N` | 各レベルで生成するエピソード数 | 10 | `--n-episodes 5` |
| `--only-level N` | 指定したレベルのみ実行（1, 2, 3） | すべて実行 | `--only-level 1` |
| `--visualize-only` | データ生成をスキップして可視化のみ | false | `--visualize-only` |

## 出力ファイル

スクリプト実行後、以下のファイルが生成されます：

```
data/minigrid/
├── long_horizon_level1_debug.npz    # Level 1データ
├── long_horizon_level2_debug.npz    # Level 2データ
└── long_horizon_level3_debug.npz    # Level 3データ

visualizations/minigrid/
├── level1/
│   ├── minigrid_episode_0_visualization.png
│   └── minigrid_episode_1_visualization.png
├── level2/
│   ├── minigrid_episode_0_visualization.png
│   └── minigrid_episode_1_visualization.png
└── level3/
    ├── minigrid_episode_0_visualization.png
    └── minigrid_episode_1_visualization.png
```

## 実行例

### クイックテスト（5分）

まず簡単なテストから始める：

```bash
# Level 1のみ、3エピソード
bash pldm_envs/minigrid/test_all_levels.sh --only-level 1 --n-episodes 3
```

### 標準テスト（15分）

各レベル5エピソードずつ：

```bash
bash pldm_envs/minigrid/test_all_levels.sh --n-episodes 5
```

### フルテスト（45分）

すべてのレベルで10エピソード：

```bash
bash pldm_envs/minigrid/test_all_levels.sh
```

## トラブルシューティング

### スクリプトが見つからない

```bash
# プロジェクトルートから実行
cd /Users/shunsei/works/MatsuoLab/WorldModel/PLDM
bash pldm_envs/minigrid/test_all_levels.sh
```

### 権限エラー

```bash
# 実行権限を付与
chmod +x pldm_envs/minigrid/test_all_levels.sh
```

### データ生成が遅い

opencv-pythonをインストールすると高速化されます：

```bash
pip install opencv-python
```

### メモリ不足

エピソード数を減らしてください：

```bash
bash pldm_envs/minigrid/test_all_levels.sh --n-episodes 3
```

## スクリプトの内部処理

1. **環境セットアップ**
   - 仮想環境のアクティベート
   - PYTHONPATHの設定
   - 必要なディレクトリの作成

2. **各レベルごとに実行**
   - データ生成（[generate_data.py](data_generation/generate_data.py)）
   - エピソード0の可視化（[visualize_episode.py](visualize_episode.py)）
   - エピソード1の可視化

3. **結果サマリー**
   - 生成されたファイルのリスト表示
   - 次のステップの提案

## 次のステップ

スクリプト実行後：

1. **可視化を確認**
   ```bash
   open visualizations/minigrid/level1/minigrid_episode_0_visualization.png
   ```

2. **トレーニングデータを生成**
   - [QUICKSTART_LONG_HORIZON.md](QUICKSTART_LONG_HORIZON.md) のコマンドを参照
   - 20,000エピソードの本番データを生成

3. **モデルを学習**
   ```bash
   python pldm/train.py --config pldm_envs/minigrid/configs/long_horizon_level1.yaml
   ```

## 関連ドキュメント

- [QUICKSTART_LONG_HORIZON.md](QUICKSTART_LONG_HORIZON.md) - クイックスタートガイド
- [LONG_HORIZON.md](LONG_HORIZON.md) - 詳細なドキュメント
- [README.md](README.md) - MiniGrid環境の基本的な使い方
