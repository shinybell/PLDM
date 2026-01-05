# DiscreteJEPA実装ガイド

FSQ (Finite Scalar Quantization)を用いた離散潜在変数によるロングホライズン予測の検証実装

## 📋 概要

このプロジェクトは、PLDMに**DiscreteJEPA**を追加実装したものです。FSQを用いて潜在表現を離散化し、離散空間でのロングホライズン予測精度を検証します。

### アーキテクチャ

```
画像 → Encoder → 連続表現 (z_continuous)
                   ↓
              [VICReg損失を適用]
                   ↓
               FSQ量子化 → 離散インデックス (z_indices)
                   ↓
              Re-embedding → 離散表現 (z_discrete)
                   ↓
    Predictor (z_discrete + action → 次のz_indices)
                   ↓
           [CrossEntropy損失を適用]
```

## 🗂️ ファイル構成

```
pldm/
├── configs_discrete.py              # DiscreteJEPA用の設定クラス
├── train_discrete.py                # DiscreteJEPA訓練スクリプト
├── models/
│   ├── quantizers.py                # FSQ量子化モジュール
│   ├── discrete_jepa.py             # DiscreteJEPAモデル本体
│   ├── discrete_predictors.py       # 離散Predictor
│   └── predictors.py                # (既存に離散版を統合)
├── objectives/
│   └── discrete_objectives.py       # VICReg + CrossEntropy損失
└── configs/minigrid/
    ├── level1_annotated.yaml        # 既存（連続JEPA用）
    └── level1_discrete.yaml         # 新規（DiscreteJEPA用）
```

## 🚀 使い方

### 1. 基本的な訓練

```bash
python pldm/train_discrete.py --configs pldm/configs/minigrid/level1_discrete.yaml
```

### 2. WandBを有効化

```bash
python pldm/train_discrete.py \
  --configs pldm/configs/minigrid/level1_discrete.yaml \
  --values wandb=True
```

### 3. FSQレベルを変更（コードブックサイズ調整）

```bash
# より小さいコードブック (512個)
python pldm/train_discrete.py \
  --configs pldm/configs/minigrid/level1_discrete.yaml \
  --values discrete_hjepa.level1.fsq.levels=[8,8,8]

# より大きいコードブック (1,000,000個)
python pldm/train_discrete.py \
  --configs pldm/configs/minigrid/level1_discrete.yaml \
  --values discrete_hjepa.level1.fsq.levels=[10,10,10,10,10,10]
```

### 4. Teacher Forcing比率を調整

```bash
python pldm/train_discrete.py \
  --configs pldm/configs/minigrid/level1_discrete.yaml \
  --values discrete_hjepa.level1.predictor.teacher_forcing_ratio=0.5
```

### 5. デバッグモード

```bash
python pldm/train_discrete.py \
  --configs pldm/configs/minigrid/level1_discrete.yaml \
  --values quick_debug=True epochs=3
```

## 🔧 主要コンポーネント

### 1. FSQ量子化（`quantizers.py`）

```python
from pldm.models.quantizers import FSQ, FSQEmbedding

# FSQ初期化
fsq = FSQ(
    levels=[8, 8, 8, 5, 5, 5],  # 各次元のレベル数
    dim=512,                     # 入力次元
    num_codebooks=6,             # コードブック次元
)

# 量子化
output = fsq(z_continuous)
z_quantized = output.quantized  # 量子化値
z_indices = output.indices      # 離散インデックス

# Re-embedding
fsq_embedding = FSQEmbedding(levels=[8, 8, 8, 5, 5, 5], embed_dim=512)
z_discrete = fsq_embedding(z_indices)
```

### 2. DiscreteJEPA（`discrete_jepa.py`）

```python
from pldm.models.discrete_jepa import DiscreteJEPA

model = DiscreteJEPA(
    config=config.discrete_hjepa.level1,
    input_dim=(3, 72, 72),
)

# 訓練時
forward_result = model.forward_posterior(
    input_states=observations,  # (T, B, C, H, W)
    actions=actions,            # (T-1, B, A)
)

# 推論時
forward_result = model.forward_prior(
    input_states=initial_state,  # (B, C, H, W)
    actions=actions,             # (T, B, A)
    T=16,
)
```

### 3. 損失関数（`discrete_objectives.py`）

```python
from pldm.objectives.discrete_objectives import build_discrete_objectives

objectives = build_discrete_objectives(
    config=config.objectives_discrete,
    repr_dim=512,
    fsq_levels=[8, 8, 8, 5, 5, 5],
)

losses = objectives(forward_result)
# losses['vicreg_loss']              - FSQ前の連続表現用
# losses['discrete_prediction_loss'] - 離散予測用（CrossEntropy）
# losses['total_loss']               - 合計
```

## 📊 設定パラメータ詳細

### FSQ設定

```yaml
discrete_hjepa:
  level1:
    fsq:
      levels: [8, 8, 8, 5, 5, 5]  # コードブックサイズ = 8³×5³ = 64,000
      num_codebooks: 6
      project_in: true             # 入力を射影
      eps: 1.0e-5                  # 数値安定性
```

**推奨コードブックサイズ:**
- 小規模: `[8, 8, 8]` → 512個
- 標準: `[8, 8, 8, 5, 5, 5]` → 64,000個
- 大規模: `[10, 10, 10, 10]` → 10,000個

### Predictor設定

```yaml
discrete_hjepa:
  level1:
    predictor:
      predictor_arch: discrete_rnn
      rnn_hidden_dim: 512
      rnn_layers: 1
      teacher_forcing_ratio: 1.0  # 訓練初期は1.0推奨
```

### 損失関数設定

```yaml
objectives_discrete:
  vicreg:
    sim_coeff: 1.0      # 類似性損失
    std_coeff: 3.0      # 標準偏差損失
    cov_coeff: 6.9238   # 共分散損失

  discrete_prediction:
    weight: 1.0         # VICRegとのバランス
    label_smoothing: 0.0
```

## 🔬 実験方法

### 1. ベースライン訓練（連続JEPA）

```bash
python pldm/train.py --configs pldm/configs/minigrid/level1_annotated.yaml
```

### 2. DiscreteJEPA訓練

```bash
python pldm/train_discrete.py --configs pldm/configs/minigrid/level1_discrete.yaml
```

### 3. 比較評価

両方のモデルで以下を測定:
- **短期予測精度** (1-4ステップ): 連続 vs 離散
- **長期予測精度** (8-16ステップ): 離散化の効果
- **コードブック利用率**: 離散表現の多様性
- **プランニング成功率**: 実タスクでの性能

## 📈 メトリクス

訓練中に以下のメトリクスが記録されます:

```python
metrics = {
    'train/loss': 総損失,
    'train/vicreg_loss': VICReg損失,
    'train/discrete_loss': CrossEntropy損失,
    'train/avg_accuracy': 予測正解率,
    'train/codebook_usage': コードブック利用率,
    'train/lr': 学習率,
}
```

## 🐛 トラブルシューティング

### Q1: メモリ不足

```bash
# バッチサイズを小さくする
python pldm/train_discrete.py \
  --configs pldm/configs/minigrid/level1_discrete.yaml \
  --values data.minigrid_config.batch_size=8
```

### Q2: 学習が不安定

```bash
# 学習率を下げる
python pldm/train_discrete.py \
  --configs pldm/configs/minigrid/level1_discrete.yaml \
  --values base_lr=0.0005
```

### Q3: コードブック崩壊

FSQはコードブック崩壊が起きにくい設計ですが、もし発生したら:
- FSQレベル数を調整: `levels=[8,8,8]` など小さめにする
- Teacher forcing比率を上げる: `teacher_forcing_ratio=1.0`

### Q4: 予測精度が低い

```bash
# RNN層を増やす
python pldm/train_discrete.py \
  --configs pldm/configs/minigrid/level1_discrete.yaml \
  --values discrete_hjepa.level1.predictor.rnn_layers=2

# 隠れ層を大きくする
python pldm/train_discrete.py \
  --configs pldm/configs/minigrid/level1_discrete.yaml \
  --values discrete_hjepa.level1.predictor.rnn_hidden_dim=1024
```

## 📚 参考文献

- **FSQ論文**: [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505) (ICLR 2024)
- **DiscreteJEPA**: 参考にしたアーキテクチャ [arXiv:2506.14373](https://arxiv.org/html/2506.14373v2)
- **PLDM**: 元の実装 [GitHub](https://latent-planning.github.io/)

## 🔄 既存コードとの互換性

DiscreteJEPAは既存のPLDMコードと完全に独立しています:
- **設定ファイル**: 別ファイル（`level1_discrete.yaml`）
- **訓練スクリプト**: 別スクリプト（`train_discrete.py`）
- **モデルクラス**: 別クラス（`DiscreteJEPA`）

既存の連続JEPAに影響を与えずに使用できます。

## ✅ 実装完了チェックリスト

- [x] FSQ量子化モジュール (`quantizers.py`)
- [x] 離散Predictor (`discrete_predictors.py`)
- [x] DiscreteJEPAモデル (`discrete_jepa.py`)
- [x] 損失関数 (`discrete_objectives.py`)
- [x] 設定クラス (`configs_discrete.py`)
- [x] 訓練スクリプト (`train_discrete.py`)
- [x] YAML設定ファイル (`level1_discrete.yaml`)

## 🎯 次のステップ

1. **データ準備**: MiniGridデータセットを生成
2. **ベースライン訓練**: 連続JEPAで性能を確認
3. **DiscreteJEPA訓練**: 離散版を訓練
4. **比較実験**: ロングホライズン予測精度を比較
5. **分析**: コードブック利用率、予測正解率などを分析

---

## 💡 Tips

- **Teacher Forcing**: 訓練初期は1.0、後期は徐々に減らす
- **コードブックサイズ**: 環境の複雑さに応じて調整（MiniGridは64K程度が適切）
- **評価頻度**: 長期訓練時は`eval_every_n_epochs`を大きめに設定してコスト削減

質問や問題があれば、実装者にお問い合わせください！
