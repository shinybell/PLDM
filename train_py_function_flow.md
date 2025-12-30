# train.py の main 関数実行時の関数呼び出しフロー

## エントリーポイント

### [train.py:565-567](pldm/train.py#L565-L567)
```python
if __name__ == "__main__":
    cfg = TrainConfig.parse_from_command_line()
    main(cfg)
```

---

## 1. 設定の解析 (Configuration Parsing)

### 1.1 TrainConfig.parse_from_command_line()
- **ファイル**: [pldm/configs.py:20-21](pldm/configs.py#L20-L21)
- **クラス**: `ConfigBase`
- **関数**: `parse_from_command_line()`
  - 内部で `omegaconf_parse(cls)` を呼び出し

### 1.2 omegaconf_parse()
- **ファイル**: [pldm/configs.py:216-232](pldm/configs.py#L216-L232)
- **機能**: コマンドライン引数を解析
  - `--configs`: 設定ファイルのリスト
  - `--values`: ドット記法での設定値の変更
- **呼び出し**: `omegaconf_parse_files_vals(cls, args.configs, args.values)`

### 1.3 omegaconf_parse_files_vals()
- **ファイル**: [pldm/configs.py:235-242](pldm/configs.py#L235-L242)
- **機能**: 複数の設定ファイルとコマンドライン値をマージ
  - OmegaConf を使用して設定を統合
  - `cls.parse_from_dict()` でデータクラスに変換

---

## 2. main 関数の実行

### 2.1 main(config: TrainConfig)
- **ファイル**: [pldm/train.py:552-562](pldm/train.py#L552-L562)
- **処理フロー**:
  1. `torch.set_num_threads(1)` - スレッド数を設定
  2. `Trainer(config)` - Trainer インスタンスを作成
  3. 条件分岐:
     - `config.eval_only` かつ `not config.quick_debug`: `trainer.validate()` を実行
     - それ以外: `trainer.train()` を実行
     - `config.quick_debug`: `trainer.validate()` を実行

---

## 3. Trainer の初期化

### 3.1 Trainer.__init__(config: TrainConfig)
- **ファイル**: [pldm/train.py:153-268](pldm/train.py#L153-L268)

#### 3.1.1 ロガーの初期化
- **関数**: `Logger.run().initialize()`
- **ファイル**: [pldm/logger.py:34-69](pldm/logger.py#L34-L69)
- **機能**:
  - シングルトンパターンでロガーを取得
  - 出力ディレクトリの作成
  - Weights & Biases (wandb) の初期化（有効な場合）
  - 設定の保存

#### 3.1.2 シード設定
- **関数**: `seed_everything(config.seed)`
- **ファイル**: [pldm/train.py:43-46](pldm/train.py#L43-L46)
- **機能**: PyTorch, NumPy, Python の乱数シードを設定

#### 3.1.3 データセットの作成
- **クラス**: `DatasetFactory`
- **ファイル**: [pldm/data/dataset_factory.py:23-46](pldm/data/dataset_factory.py#L23-L46)
- **関数**: `create_datasets()`
  - データセットタイプに応じて以下のいずれかを呼び出し:
    - `_create_single_datasets()` - Dot データセット
    - `_create_wall_datasets()` - Wall データセット
    - `_create_wall_expert_datasets()` - Wall Expert データセット
    - `_create_d4rl_datasets()` - D4RL データセット
    - `_create_locomaze_datasets()` - LocoMaze データセット

**データセット作成の詳細**:
- **Wall データセット**の場合 ([dataset_factory.py:59-80](pldm/data/dataset_factory.py#L59-L80)):
  - オフライン使用時: `OfflineWallDataset` + `make_dataloader()`
  - オンライン使用時: `WallDataset` + `make_dataloader_for_prebatched_ds()`
  - プロービング用データセット: `_create_wall_probing_datasets()`

#### 3.1.4 モデルの作成
- **クラス**: `HJEPA`
- **ファイル**: pldm/models/hjepa.py
- **入力**:
  - `config.hjepa`: HJEPA の設定
  - `input_dim`: データから推論された入力次元
  - `normalizer`: データセットの正規化器
  - `use_propio_pos`, `use_propio_vel`: 固有受容的状態の使用フラグ

#### 3.1.5 目的関数の構築
- **関数**: `config.objectives_l1.build_objectives_list()`
- **ファイル**: [pldm/objectives/__init__.py:31-86](pldm/objectives/__init__.py#L31-L86)
- **機能**: 設定された目的関数タイプに応じて目的関数リストを作成
  - VICRegObjective
  - IDMObjective
  - PredictionObjective
  - など

#### 3.1.6 チェックポイントのロード（オプション）
- **関数**: `maybe_load_model()`
- **ファイル**: [pldm/train.py:285-312](pldm/train.py#L285-L312)
- **機能**: `config.load_checkpoint_path` が指定されている場合、モデルの重みをロード

#### 3.1.7 モデルのコンパイル（オプション）
- `config.compile_model` が True の場合、`torch.compile(self.model)` を実行

---

## 4. 学習の実行

### 4.1 Trainer.train()
- **ファイル**: [pldm/train.py:314-442](pldm/train.py#L314-L442)

#### 4.1.1 オプティマイザの作成
- **クラス**: `OptimizerFactory`
- **ファイル**: [pldm/optimizers/optimizer_factory.py:11-46](pldm/optimizers/optimizer_factory.py#L11-L46)
- **関数**: `create_optimizer()`
- **サポートされるオプティマイザ**:
  - LARS ([optimizer_factory.py:24-30](pldm/optimizers/optimizer_factory.py#L24-L30))
  - Adam ([optimizer_factory.py:32-42](pldm/optimizers/optimizer_factory.py#L32-L42))

#### 4.1.2 学習の再開（オプション）
- **関数**: `maybe_resume()`
- **ファイル**: [pldm/train.py:270-283](pldm/train.py#L270-L283)
- **機能**: 既存のチェックポイントから学習を再開

#### 4.1.3 学習率スケジューラの作成
- **クラス**: `Scheduler`
- **ファイル**: pldm/optimizers/schedulers.py
- **機能**: 学習率のスケジューリング

#### 4.1.4 エポックループ
**各エポックで**:

1. **バッチループ**: データローダーから各バッチを取得

2. **データの前処理**:
   - CUDA へ転送
   - バッチ次元と時間次元の入れ替え

3. **学習率の調整**:
   - `scheduler.adjust_learning_rate(step)`

4. **順伝播**:
   - **関数**: `model.forward_posterior(s, a, **optional_fields)`
   - **ファイル**: pldm/models/hjepa.py

5. **損失計算**:
   - 各目的関数に対して:
     - `objective(batch, [forward_result.level1])`
   - 総損失: `sum([loss_info.total_loss for loss_info in loss_infos])`

6. **逆伝播と最適化**:
   - `total_loss.backward()`
   - `optimizer.step()`

7. **EMA 更新**:
   - `model.update_ema()`

8. **ログ記録** (100 ステップごと):
   - **関数**: `Logger.run().log()`
   - メトリクスを wandb とローカルファイルに記録

#### 4.1.5 チェックポイント保存
- **関数**: `save_model()`
- **ファイル**: [pldm/train.py:534-549](pldm/train.py#L534-L549)
- **タイミング**: `config.save_every_n_epochs` エポックごと

#### 4.1.6 検証
- **関数**: `validate()`
- **タイミング**: `config.eval_every_n_epochs` エポックごと

---

## 5. 評価の実行

### 5.1 Trainer.validate()
- **ファイル**: [pldm/train.py:494-532](pldm/train.py#L494-L532)

#### 5.1.1 目的関数での評価
- **関数**: `eval_on_objectives()`
- **ファイル**: [pldm/train.py:445-492](pldm/train.py#L445-L492)
- **機能**: 検証データセットで損失を計算

#### 5.1.2 Evaluator の作成
- **クラス**: `Evaluator`
- **ファイル**: pldm/evaluation/evaluator.py
- **機能**: プロービングとプランニングの評価

#### 5.1.3 評価の実行
- **関数**: `evaluator.evaluate()`
- **機能**:
  - プロービング評価
  - プランニング評価
  - 結果のログ記録と保存

---

## 主要な依存モジュール

### データ関連
- `pldm/data/dataset_factory.py` - データセットの作成
- `pldm/data/enums.py` - データ設定の列挙型
- `pldm/data/utils.py` - データユーティリティ関数
- `pldm_envs/*` - 環境固有のデータセット実装

### モデル関連
- `pldm/models/hjepa.py` - HJEPA モデルの実装

### 目的関数関連
- `pldm/objectives/__init__.py` - 目的関数の設定
- `pldm/objectives/vicreg.py` - VICReg 目的関数
- `pldm/objectives/idm.py` - IDM 目的関数
- `pldm/objectives/prediction.py` - 予測目的関数

### 最適化関連
- `pldm/optimizers/optimizer_factory.py` - オプティマイザの作成
- `pldm/optimizers/schedulers.py` - 学習率スケジューラ
- `pldm/optimizers/lars.py` - LARS オプティマイザ

### 評価関連
- `pldm/evaluation/evaluator.py` - 評価の実行
- `pldm/probing/evaluator.py` - プロービング評価

### ユーティリティ
- `pldm/logger.py` - ロギング機能
- `pldm/configs.py` - 設定の解析
- `pldm/utils.py` - 汎用ユーティリティ

---

## 実行フローサマリー

```
main()
├── TrainConfig.parse_from_command_line()
│   └── omegaconf_parse()
│       └── omegaconf_parse_files_vals()
│
└── Trainer(config)
    ├── __init__()
    │   ├── Logger.run().initialize()
    │   ├── seed_everything()
    │   ├── DatasetFactory().create_datasets()
    │   ├── HJEPA() (モデル作成)
    │   ├── config.objectives_l1.build_objectives_list()
    │   ├── maybe_load_model()
    │   └── torch.compile() (オプション)
    │
    ├── train() (eval_only でない場合)
    │   ├── OptimizerFactory().create_optimizer()
    │   ├── maybe_resume()
    │   ├── Scheduler()
    │   └── エポックループ
    │       ├── バッチループ
    │       │   ├── scheduler.adjust_learning_rate()
    │       │   ├── model.forward_posterior()
    │       │   ├── objective() (各目的関数)
    │       │   ├── backward() & step()
    │       │   ├── model.update_ema()
    │       │   └── Logger.run().log()
    │       ├── save_model() (定期的)
    │       └── validate() (定期的)
    │
    └── validate() (eval_only の場合)
        ├── eval_on_objectives()
        └── Evaluator().evaluate()
```
