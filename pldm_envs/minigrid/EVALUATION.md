# MiniGrid環境での評価と可視化

学習済みPLDMモデルの評価と可視化を行う方法を説明します。

## 必要なパッケージ

```bash
pip install imageio imageio-ffmpeg pillow
```

## 1. モデルの評価

### 基本的な使い方

学習済みモデルで複数エピソードを実行し、成功率やエピソード長などを測定します。

```bash
python pldm_envs/minigrid/evaluate_model.py \
  --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 100 \
  --obs_size 72
```

### オプション

- `--checkpoint`: モデルのチェックポイントファイルパス（必須）
- `--config`: 設定ファイルパス（オプション、チェックポイントに含まれていない場合）
- `--env_name`: 評価する環境名（デフォルト: `MiniGrid-LongHorizon-Level1-v0`）
- `--n_episodes`: 評価エピソード数（デフォルト: 100）
- `--obs_size`: 観測画像サイズ（デフォルト: 72）
- `--max_steps`: エピソードの最大ステップ数（デフォルト: 1000）
- `--device`: 使用するデバイス（`cpu` or `cuda`）
- `--output`: 結果をJSONファイルとして保存（オプション）

### 出力例

```
============================================================
Evaluation Results
============================================================
Success Rate: 45.00% ± 49.75%
Episode Length: 256.3 ± 125.4
Episode Reward: 0.450 ± 0.498
============================================================
```

### 結果の保存

```bash
python pldm_envs/minigrid/evaluate_model.py \
  --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 100 \
  --obs_size 72 \
  --output results/level1_evaluation.json
```

## 2. エージェントの可視化

### GIFとして保存

単一エピソードをGIFアニメーションとして保存します。

```bash
python pldm_envs/minigrid/visualize_agent.py \
  --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --output visualizations/agent_episode.gif \
  --obs_size 72 \
  --fps 10
```

### ビデオとして保存

複数エピソードをMP4ビデオとして保存します。

```bash
python pldm_envs/minigrid/visualize_agent.py \
  --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --output visualizations/agent_episodes.mp4 \
  --n_episodes 5 \
  --obs_size 72 \
  --fps 10
```

### 個別画像として保存

フレームごとに個別のPNG画像として保存します。

```bash
python pldm_envs/minigrid/visualize_agent.py \
  --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --output visualizations/frames/ \
  --obs_size 72
```

### オプション

- `--checkpoint`: モデルのチェックポイントファイルパス（必須）
- `--config`: 設定ファイルパス（オプション）
- `--env_name`: 環境名
- `--output`: 出力先（`.gif`, `.mp4`, またはディレクトリパス）（必須）
- `--n_episodes`: 可視化するエピソード数（デフォルト: 1）
- `--obs_size`: 観測画像サイズ（デフォルト: 72）
- `--max_steps`: 最大ステップ数（デフォルト: 1000）
- `--fps`: ビデオ/GIFのフレームレート（デフォルト: 10）
- `--action_mode`: アクション選択モード
  - `greedy`: 常に最良のアクションを選択
  - `epsilon-greedy`: ε確率でランダムアクション（デフォルト）
  - `random`: 完全ランダム
- `--epsilon`: epsilon-greedyのε値（デフォルト: 0.1）
- `--device`: 使用するデバイス（`cpu` or `cuda`）

## 3. 異なるレベルでの評価

### Level 1 (8x8グリッド)

```bash
python pldm_envs/minigrid/evaluate_model.py \
  --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 100 \
  --obs_size 72
```

### Level 2 (16x16グリッド)

```bash
python pldm_envs/minigrid/evaluate_model.py \
  --checkpoint checkpoints/minigrid_level2/latest.ckpt \
  --env_name MiniGrid-LongHorizon-Level2-v0 \
  --n_episodes 100 \
  --obs_size 72
```

### Level 3 (24x24グリッド)

```bash
python pldm_envs/minigrid/evaluate_model.py \
  --checkpoint checkpoints/minigrid_level3/latest.ckpt \
  --env_name MiniGrid-LongHorizon-Level3-v0 \
  --n_episodes 100 \
  --obs_size 72
```

## 4. バッチ評価スクリプト例

複数のチェックポイントを一括評価するシェルスクリプト例：

```bash
#!/bin/bash

CHECKPOINTS=(
    "checkpoints/minigrid_level1_test/epoch=0_sample_step=2000.ckpt"
    "checkpoints/minigrid_level1_test/epoch=1_sample_step=4000.ckpt"
    "checkpoints/minigrid_level1_test/epoch=2_sample_step=6000.ckpt"
    "checkpoints/minigrid_level1_test/latest.ckpt"
)

for ckpt in "${CHECKPOINTS[@]}"; do
    echo "Evaluating $ckpt"
    python pldm_envs/minigrid/evaluate_model.py \
        --checkpoint "$ckpt" \
        --env_name MiniGrid-LongHorizon-Level1-v0 \
        --n_episodes 100 \
        --obs_size 72 \
        --output "results/$(basename $ckpt .ckpt).json"
done
```

## 5. トラブルシューティング

### エラー: `ModuleNotFoundError: No module named 'imageio'`

```bash
pip install imageio imageio-ffmpeg
```

### エラー: チェックポイントが見つからない

チェックポイントのパスを確認してください：

```bash
ls checkpoints/minigrid_level1_test/
```

最新のチェックポイントは `latest.ckpt` として保存されています。

### 警告: 成功率が0%

モデルがまだ十分に学習していない可能性があります。以下を確認してください：

1. 学習が完了しているか（十分なエポック数）
2. チェックポイントが正しくロードされているか
3. 環境名と観測サイズが学習時と一致しているか

### GPU使用時のメモリエラー

`--device cpu` を使用してCPUで実行してください：

```bash
python pldm_envs/minigrid/evaluate_model.py \
  --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
  --device cpu \
  --n_episodes 100
```

## 6. 高度な使用例

### カスタムアクション選択戦略

完全に貪欲な戦略で評価：

```bash
python pldm_envs/minigrid/visualize_agent.py \
  --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
  --output visualizations/greedy.gif \
  --action_mode greedy
```

### ランダムベースラインとの比較

```bash
# 学習済みモデル
python pldm_envs/minigrid/evaluate_model.py \
  --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
  --n_episodes 100 \
  --output results/trained_model.json

# ランダムエージェント（可視化スクリプトでランダムモード使用）
python pldm_envs/minigrid/visualize_agent.py \
  --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
  --output visualizations/random.gif \
  --action_mode random
```

## 注意事項

1. **アクション選択について**: 現在の実装は簡易的なもので、次状態予測の変化量をスコアとしています。より高度な方法（価値関数、Q学習など）を実装すると性能が向上する可能性があります。

2. **評価の安定性**: MiniGridはランダムに迷路が生成されるため、評価結果には分散があります。十分な数のエピソード（100以上）で評価することを推奨します。

3. **可視化のパフォーマンス**: 長いエピソードや多数のエピソードを可視化する場合、メモリ使用量が大きくなる可能性があります。

## 参考資料

- [MiniGrid公式ドキュメント](https://minigrid.farama.org/)
- [QUICKSTART_LONG_HORIZON.md](QUICKSTART_LONG_HORIZON.md) - MiniGrid環境の詳細
- [PLDM_INTEGRATION.md](PLDM_INTEGRATION.md) - PLDM学習の詳細
