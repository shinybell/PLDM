# MiniGrid 並列データ生成ガイド

MiniGridのデータ生成を並列化して高速化する方法を説明します。

## 概要

ワーカー分割方式を使用して、複数のプロセスで並列にエピソードを生成します。

- **方法1: シェルスクリプトを使用（推奨）** - 自動的に並列実行と結合を行う
- **方法2: 手動で並列実行** - 各ワーカーを個別に起動

## 方法1: シェルスクリプトを使用（推奨）

### 基本的な使い方

```bash
# 4ワーカーで10,000エピソードを生成
bash pldm_envs/minigrid/data_generation/parallel_generate.sh \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 10000 \
    --workers 4 \
    --output_dir data/minigrid/empty_8x8
```

### オプション指定

```bash
# 画像リサイズとパディングを含む例
bash pldm_envs/minigrid/data_generation/parallel_generate.sh \
    --env_name MiniGrid-DoorKey-8x8-v0 \
    --n_episodes 20000 \
    --workers 8 \
    --output_dir data/minigrid/doorkey \
    --max_steps 200 \
    --resize 72 \
    --pad_length 201
```

### 実行フロー

1. **並列実行**: 各ワーカーが独立して一部のエピソードを生成
   - Worker 0: episodes 0-2499
   - Worker 1: episodes 2500-4999
   - Worker 2: episodes 5000-7499
   - Worker 3: episodes 7500-9999

2. **自動結合**: すべてのワーカーが完了後、自動的にデータを結合

3. **クリーンアップ**: オプションでワーカーファイルを削除

### 進捗確認

別のターミナルで以下のコマンドを実行して進捗を確認できます:

```bash
# すべてのワーカーのログを監視
tail -f data/minigrid/empty_8x8/worker_*.log

# 特定のワーカーのログを確認
tail -f data/minigrid/empty_8x8/worker_0.log
```

## 方法2: 手動で並列実行

### ステップ1: 各ワーカーを個別に起動

異なるターミナルで各ワーカーを起動するか、`&`でバックグラウンド実行:

```bash
# Worker 0
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 10000 \
    --output_path data/minigrid/worker_0.npz \
    --workers_num 4 \
    --worker_id 0 &

# Worker 1
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 10000 \
    --output_path data/minigrid/worker_1.npz \
    --workers_num 4 \
    --worker_id 1 &

# Worker 2, 3も同様に...
```

### ステップ2: データセットを結合

すべてのワーカーが完了したら、データを結合:

```bash
python pldm_envs/minigrid/data_generation/merge_datasets.py \
    --input_pattern "data/minigrid/worker_*.npz" \
    --output_path data/minigrid/train.npz
```

または、特定のファイルを指定:

```bash
python pldm_envs/minigrid/data_generation/merge_datasets.py \
    --input_files data/minigrid/worker_0.npz \
                  data/minigrid/worker_1.npz \
                  data/minigrid/worker_2.npz \
                  data/minigrid/worker_3.npz \
    --output_path data/minigrid/train.npz
```

## パフォーマンス

### 並列化の効果

- **シングルプロセス**: 10,000エピソード ≈ 30分
- **4ワーカー**: 10,000エピソード ≈ 8分（約3.75倍高速）
- **8ワーカー**: 10,000エピソード ≈ 4分（約7.5倍高速）

実際の速度向上はCPUコア数に依存します。

### ワーカー数の選択

推奨されるワーカー数:

```bash
# CPUコア数を確認
# macOS/Linux
nproc  # または sysctl -n hw.ncpu

# 推奨: (CPUコア数 - 1) または (CPUコア数)
# 例: 8コアCPU → 4-8ワーカー
```

## 注意事項

### エピソード数の制約

**重要**: `n_episodes`は`workers_num`で割り切れる必要があります。

```bash
# ✅ OK: 10000 % 4 = 0
--n_episodes 10000 --workers 4

# ❌ エラー: 10000 % 3 ≠ 0
--n_episodes 10000 --workers 3

# ✅ OK: 9000 % 3 = 0
--n_episodes 9000 --workers 3
```

### メモリ使用量

各ワーカーは独立した環境とデータバッファを持つため、メモリ使用量は以下のようになります:

```
総メモリ使用量 ≈ (1ワーカーのメモリ) × (ワーカー数)
```

メモリが不足する場合はワーカー数を減らしてください。

## トラブルシューティング

### エラー: "n_episodes must be divisible by workers_num"

エピソード数をワーカー数で割り切れる値に調整してください:

```bash
# 例: 10007エピソードを8ワーカーで実行したい
# → 10000 (8で割り切れる) または 10008に調整
```

### ワーカーが失敗する

ログを確認:

```bash
cat data/minigrid/empty_8x8/worker_0.log
```

一般的な原因:
- メモリ不足 → ワーカー数を減らす
- 環境名の誤り → `--env_name`を確認
- ディスク容量不足 → 空き容量を確保

### 結合が失敗する

ファイルが正しく生成されているか確認:

```bash
ls -lh data/minigrid/worker_*.npz

# 各ファイルのサイズが0でないことを確認
```

## 使用例

### 例1: 小規模データセット（デバッグ用）

```bash
bash pldm_envs/minigrid/data_generation/parallel_generate.sh \
    --env_name MiniGrid-Empty-8x8-v0 \
    --n_episodes 1000 \
    --workers 2 \
    --output_dir data/minigrid/debug
```

### 例2: 中規模データセット

```bash
bash pldm_envs/minigrid/data_generation/parallel_generate.sh \
    --env_name MiniGrid-DoorKey-16x16-v0 \
    --n_episodes 50000 \
    --workers 8 \
    --output_dir data/minigrid/doorkey_50k \
    --max_steps 300 \
    --resize 72
```

### 例3: 大規模データセット（Long Horizon環境）

```bash
bash pldm_envs/minigrid/data_generation/parallel_generate.sh \
    --env_name MiniGrid-LongHorizon-Level1-v0 \
    --n_episodes 100000 \
    --workers 16 \
    --output_dir data/minigrid/long_horizon_100k \
    --max_steps 500 \
    --resize 72 \
    --pad_length 501
```

## 参考

- [generate_data.py](generate_data.py) - 基本的なデータ生成スクリプト
- [merge_datasets.py](merge_datasets.py) - データセット結合スクリプト
- [parallel_generate.sh](parallel_generate.sh) - 並列実行スクリプト
