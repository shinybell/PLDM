# データセット作成の詳細フロー

## 概要

`train.py`の`Trainer.__init__()`で呼び出される`DatasetFactory`がデータセット作成を担当します。データセットタイプに応じて異なるデータセットが作成されます。

---

## 1. DatasetFactory の初期化と呼び出し

### 1.1 DatasetFactory のインスタンス化
- **ファイル**: [pldm/train.py:172-176](pldm/train.py#L172-L176)
```python
datasets = DatasetFactory(
    config.data,
    probing_cfg=config.eval_cfg.probing,
    disable_l2=config.hjepa.disable_l2,
).create_datasets()
```

### 1.2 DatasetFactory.__init__()
- **ファイル**: [pldm/data/dataset_factory.py:24-32](pldm/data/dataset_factory.py#L24-L32)
- **パラメータ**:
  - `config: DataConfig` - データセット設定
  - `probing_cfg: ProbingConfig` - プロービング評価用の設定
  - `disable_l2: bool` - レベル2を無効化するフラグ

---

## 2. create_datasets() - データセットタイプの分岐

### 2.1 関数概要
- **ファイル**: [pldm/data/dataset_factory.py:34-46](pldm/data/dataset_factory.py#L34-L46)
- **機能**: `config.dataset_type`に基づいて適切なデータセット作成関数を呼び出し

### 2.2 データセットタイプ
- **定義**: [pldm/data/enums.py:14-21](pldm/data/enums.py#L14-L21)
```python
class DatasetType(Enum):
    Single = auto()      # シンプルなドット環境
    Multiple = auto()    # 複数ドット環境（未使用）
    Wall = auto()        # 壁のある環境
    WallExpert = auto()  # エキスパートデモによる壁環境
    D4RL = auto()        # D4RLベンチマーク
    D4RLEigf = auto()    # D4RL EIGF（未使用）
    LocoMaze = auto()    # LocoMaze環境
```

---

## 3. 各データセットタイプの詳細

## 3.1 Single データセット (DotDataset)

### 3.1.1 作成フロー
- **関数**: `_create_single_datasets()`
- **ファイル**: [pldm/data/dataset_factory.py:48-57](pldm/data/dataset_factory.py#L48-L57)

**処理ステップ**:
1. **訓練データセット作成**:
   - `DotDataset(self.config.dot_config)`
   - **ファイル**: [pldm_envs/wall/data/single.py:43-62](pldm_envs/wall/data/single.py#L43-L62)

2. **検証データセット作成**:
   - 同じ設定で`train=False`にして作成
   - 訓練データのnormalizerを共有

3. **戻り値**: `Datasets(ds=ds, val_ds=val_ds)`

### 3.1.2 DotDataset の詳細
- **クラス**: `DotDataset`
- **ファイル**: [pldm_envs/wall/data/single.py:43](pldm_envs/wall/data/single.py#L43)

**設定パラメータ** ([single.py:18-41](pldm_envs/wall/data/single.py#L18-L41)):
```python
@dataclass
class DotDatasetConfig:
    size: int = 10000              # データセットサイズ
    batch_size: int = 128          # バッチサイズ
    dot_std: float = 1.3           # ドットの標準偏差
    action_noise: float = 0.2      # アクションノイズ
    n_steps: int = 91              # タイムステップ数
    img_size: int = 64             # 画像サイズ
    # その他のパラメータ...
```

**主要メソッド**:
- `__len__()`: データセット長を返す (`size // batch_size`)
- `__getitem__(i)`: `generate_multistep_sample()`を呼び出し
- `generate_state_and_actions()`: 状態とアクションを生成 ([single.py:91-111](pldm_envs/wall/data/single.py#L91-L111))
- `render_location()`: 位置座標を画像に変換 ([single.py:63-89](pldm_envs/wall/data/single.py#L63-L89))

**出力データ** ([single.py:11-15](pldm_envs/wall/data/single.py#L11-L15)):
```python
class Sample(NamedTuple):
    states: torch.Tensor      # [(batch_size), T, 1, H, W]
    locations: torch.Tensor   # [(batch_size), T, N_DOTS, 2]
    actions: torch.Tensor     # [(batch_size), T, 2]
    bias_angle: torch.Tensor  # [(batch_size), 2]
```

---

## 3.2 Wall データセット

### 3.2.1 作成フロー
- **関数**: `_create_wall_datasets()`
- **ファイル**: [pldm/data/dataset_factory.py:59-80](pldm/data/dataset_factory.py#L59-L80)

**処理ステップ**:

1. **オフライン/オンライン分岐**:

   **オフラインの場合** (`config.offline_wall_config.use_offline == True`):
   - `OfflineWallDataset`を作成
   - **ファイル**: [pldm_envs/wall/data/offline_wall.py:74-138](pldm_envs/wall/data/offline_wall.py#L74-L138)
   - `make_dataloader()`でDataLoaderに変換

   **オンラインの場合**:
   - `WallDataset`を作成
   - **ファイル**: [pldm_envs/wall/data/wall.py:53-62](pldm_envs/wall/data/wall.py#L53-L62)
   - `make_dataloader_for_prebatched_ds()`でラッピング

2. **プロービング用データセット作成**:
   - `_create_wall_probing_datasets(ds.normalizer)`を呼び出し

3. **戻り値**: `Datasets(ds=ds, val_ds=None, probing_datasets=probing_datasets)`

### 3.2.2 WallDataset の詳細
- **クラス**: `WallDataset` (DotDatasetを継承)
- **ファイル**: [pldm_envs/wall/data/wall.py:53](pldm_envs/wall/data/wall.py#L53)

**設定パラメータ** ([wall.py:27-51](pldm_envs/wall/data/wall.py#L27-L51)):
```python
@dataclass
class WallDatasetConfig(DotDatasetConfig):
    fix_wall: bool = True              # 壁を固定するか
    wall_padding: int = 20             # 壁のパディング
    door_padding: int = 10             # ドアのパディング
    wall_width: int = 3                # 壁の幅
    door_space: int = 4                # ドアのスペース
    cross_wall_rate: float = 0.1       # 壁を超える軌跡の割合
    expert_cross_wall_rate: float = 0.0 # エキスパート軌跡で壁を超える割合
    # その他のパラメータ...
```

**追加機能**:
- `generate_actions_to_goal()`: ゴールへの直線アクション生成 ([wall.py:68-145](pldm_envs/wall/data/wall.py#L68-L145))
- `generate_cross_wall_points()`: 壁を超える地点を生成
- 壁とドアのレイアウト生成

**出力データ** ([wall.py:17-24](pldm_envs/wall/data/wall.py#L17-L24)):
```python
class WallSample(NamedTuple):
    states: torch.Tensor      # [(batch_size), T, 1, 28, 28]
    locations: torch.Tensor   # [(batch_size), T, 2]
    actions: torch.Tensor     # [(batch_size), T, 2]
    bias_angle: torch.Tensor  # [(batch_size), 2]
    wall_x: torch.Tensor      # [(batch_size), 1]
    door_y: torch.Tensor      # [(batch_size), 1]
```

### 3.2.3 OfflineWallDataset の詳細
- **クラス**: `OfflineWallDataset`
- **ファイル**: [pldm_envs/wall/data/offline_wall.py:74](pldm_envs/wall/data/offline_wall.py#L74)

**設定パラメータ** ([offline_wall.py:8-21](pldm_envs/wall/data/offline_wall.py#L8-L21)):
```python
@dataclass
class OfflineWallDatasetConfig:
    offline_data_path: str = ""  # オフラインデータのパス
    lazy_load: bool = False      # 遅延ロード（メモリ節約）
    n_steps: Optional[int] = None # エピソード長
    # その他のパラメータ...
```

**データロード**:
- **NPZ形式** ([offline_wall.py:35-71](pldm_envs/wall/data/offline_wall.py#L35-L71)):
  - 観測、アクション、位置、終端フラグを含む
  - 軌跡ごとに再整形

- **NP形式** ([offline_wall.py:23-32](pldm_envs/wall/data/offline_wall.py#L23-L32)):
  - 個別のnpyファイルからロード

**スライシング** ([offline_wall.py:92-104](pldm_envs/wall/data/offline_wall.py#L92-L104)):
- 各軌跡から複数のスライス（部分軌跡）を作成
- データセット長 = 軌跡数 × スライス/軌跡

### 3.2.4 プロービング用データセット作成
- **関数**: `_create_wall_probing_datasets()`
- **ファイル**: [pldm/data/dataset_factory.py:82-148](pldm/data/dataset_factory.py#L82-L148)

**作成するデータセット**:

1. **probe_ds**: 訓練用プロービングデータセット
   - `WallDataset`を`train=False`、`size=val_size`で作成
   - `n_steps=probing_cfg.l1_depth`に設定

2. **probe_val_ds**: 検証用プロービングデータセット
   - 同様の設定で作成

3. **extra_datasets** (オプション):
   - **wall_test** ([dataset_factory.py:117-129](pldm/data/dataset_factory.py#L117-L129)):
     - `WallPassingTestDataset` - 壁通過テスト用
   - **border_test** ([dataset_factory.py:130-142](pldm/data/dataset_factory.py#L130-L142)):
     - `BorderPassingTestDataset` - 境界通過テスト用

**戻り値**: `ProbingDatasets(ds=probe_ds, val_ds=probe_val_ds, extra_datasets=extra_datasets)`

---

## 3.3 WallExpert データセット

### 3.3.1 作成フロー
- **関数**: `_create_wall_expert_datasets()`
- **ファイル**: [pldm/data/dataset_factory.py:150-165](pldm/data/dataset_factory.py#L150-L165)

**処理ステップ**:
1. 空のWallDatasetを作成（normalizer取得のため）
2. `WrappedWallExpertDataset`で訓練データを作成
3. `WrappedWallExpertDataset`で検証データを作成（`train=False`）

**特徴**: エキスパート軌跡（最適な行動）を含むデータセット

---

## 3.4 D4RL データセット

### 3.4.1 作成フロー
- **関数**: `_create_d4rl_datasets()`
- **ファイル**: [pldm/data/dataset_factory.py:167-211](pldm/data/dataset_factory.py#L167-L211)

**処理ステップ**:

1. **訓練データセット作成**:
   - `D4RLDataset(self.config.d4rl_config)`
   - **ファイル**: [pldm_envs/diverse_maze/d4rl.py:20](pldm_envs/diverse_maze/d4rl.py#L20)
   - `make_dataloader()`でラッピング

2. **プロービング用訓練データセット**:
   - `path`と`images_path`を`probing_cfg`から取得
   - `sample_length=probing_cfg.l1_depth`に設定
   - `make_dataloader()`でラッピング

3. **プロービング用検証データセット**:
   - `train=False`、`crop_length=50000`に設定
   - `make_dataloader()`でラッピング

4. **戻り値**: `Datasets(ds=ds, val_ds=None, probing_datasets=ProbingDatasets(...))`

### 3.4.2 D4RLDataset の詳細
- **クラス**: `D4RLDataset`
- **ファイル**: [pldm_envs/diverse_maze/d4rl.py:20](pldm_envs/diverse_maze/d4rl.py#L20)

**初期化処理** ([d4rl.py:21-59](pldm_envs/diverse_maze/d4rl.py#L21-L59)):
```python
if config.path is None:
    # D4RLベンチマークから直接ロード
    _prepare_ds()
elif config.mixture_expert != 0:
    # ランダムとエキスパートの混合データセット
    _prepare_mixed_ds()
else:
    # 保存済みデータセットからロード
    _prepare_saved_ds()
```

**データ準備メソッド**:

1. **_prepare_ds()** ([d4rl.py:103-129](pldm_envs/diverse_maze/d4rl.py#L103-L129)):
   - D4RLライブラリから環境とデータセットをロード
   - 軌跡ごとに分割（`steps`フィールドを使用）

2. **_prepare_saved_ds()** ([d4rl.py:131-146](pldm_envs/diverse_maze/d4rl.py#L131-L146)):
   - 保存済みのtorchファイルからロード
   - 累積長を計算（スライシング用）

3. **_prepare_mixed_ds()** ([d4rl.py:61-101](pldm_envs/diverse_maze/d4rl.py#L61-L101)):
   - 保存済みデータとD4RLデータを混合
   - `mixture_expert`パラメータでエキスパート割合を制御

**画像データ**:
- `images_path`が指定されている場合:
  - **zarr形式**: メモリにロード ([d4rl.py:35-41](pldm_envs/diverse_maze/d4rl.py#L35-L41))
  - **npy形式**: mmap（メモリマップ）でロード ([d4rl.py:42-48](pldm_envs/diverse_maze/d4rl.py#L42-L48))
  - **画像ファイル**: 変換（CenterCrop、Resize）を適用

---

## 4. DataLoader作成の詳細

### 4.1 make_dataloader()
- **ファイル**: [pldm/data/utils.py:106-140](pldm/data/utils.py#L106-L140)
- **用途**: PyTorchスタイルのデータセット（`__getitem__`を持つ）用

**処理ステップ**:

1. **PyTorch DataLoader作成**:
   ```python
   loader = torch.utils.data.DataLoader(
       ds,
       config.batch_size,
       shuffle=train,
       num_workers=loader_config.num_workers,
       drop_last=True,
       prefetch_factor=1 or None,
       pin_memory=False,
   )
   ```

2. **Normalizer構築**:
   - `loader_config.normalize == True`の場合:
     - `Normalizer.build_normalizer()`を呼び出し
     - **ファイル**: [pldm_envs/utils/normalizer.py:73-80](pldm_envs/utils/normalizer.py#L73-L80)
     - データセットから100サンプル（デバッグ時は1）を取得
     - 平均と標準偏差を計算

   - `normalize == False`の場合:
     - 恒等正規化器（何もしない）を作成

3. **NormalizedDataLoader でラッピング**:
   - **クラス**: `NormalizedDataLoader`
   - **ファイル**: [pldm/data/utils.py:87-103](pldm/data/utils.py#L87-L103)
   - イテレーション時に各バッチを正規化

### 4.2 make_dataloader_for_prebatched_ds()
- **ファイル**: [pldm/data/utils.py:143-165](pldm/data/utils.py#L143-L165)
- **用途**: すでにバッチ化されたデータセット（WallDatasetなど）用

**処理ステップ**:
1. Normalizerを構築（上記と同様）
2. `NormalizedDataLoader`で直接ラッピング（PyTorchのDataLoaderを使わない）

### 4.3 Normalizer の詳細
- **クラス**: `Normalizer`
- **ファイル**: [pldm_envs/utils/normalizer.py:38-72](pldm_envs/utils/normalizer.py#L38-L72)

**正規化対象**:
- `states`: 状態（画像または固有受容情報）
- `actions`: アクション
- `locations`: 位置座標
- `propio_pos`: 固有受容位置
- `propio_vel`: 固有受容速度

**正規化方法**:
- **標準正規化**: `(x - mean) / std`
- **Min-Max正規化** (`min_max_state=True`の場合): 画像用

**build_normalizer()** ([normalizer.py:73-179](pldm_envs/utils/normalizer.py#L73)):
1. データセットから`n_samples`個のバッチを取得
2. 各フィールドのデータを収集
3. 平均と標準偏差を計算
4. `Normalizer`インスタンスを作成して返す

### 4.4 NormalizedDataLoader
- **クラス**: `NormalizedDataLoader`
- **ファイル**: [pldm/data/utils.py:87-103](pldm/data/utils.py#L87-L103)

**機能**:
```python
def __iter__(self):
    for batch in self.dataloader:
        # 正規化を適用
        new_batch = self.normalizer.normalize_sample(batch)
        yield new_batch
```

---

## 5. optional_fields の取得

### 5.1 get_optional_fields()
- **ファイル**: [pldm/data/utils.py:6-27](pldm/data/utils.py#L6-L27)
- **呼び出し元**: [pldm/train.py:371](pldm/train.py#L371) (訓練ループ内)

**処理内容**:
```python
fields = [
    "propio_vel",      # 固有受容速度
    "propio_pos",      # 固有受容位置
    "chunked_locations",
    "chunked_propio_pos",
    "chunked_propio_vel",
    "goal",            # ゴール情報
]
```

各フィールドが存在する場合:
- CUDAに転送
- バッチ次元と時間次元を入れ替え（`transpose(0, 1)`）
- 辞書に追加

**戻り値**: 辞書（存在しないフィールドは`None`）

---

## 6. データフロー全体図

```
train.py: Trainer.__init__()
    │
    ├─> DatasetFactory(config.data, probing_cfg, disable_l2)
    │       │
    │       └─> create_datasets()
    │               │
    │               ├─ DatasetType.Single
    │               │   └─> _create_single_datasets()
    │               │       ├─> DotDataset(config.dot_config)
    │               │       └─> Datasets(ds, val_ds)
    │               │
    │               ├─ DatasetType.Wall
    │               │   └─> _create_wall_datasets()
    │               │       ├─ offline: OfflineWallDataset
    │               │       │   ├─> load_npz() or load_np()
    │               │       │   └─> make_dataloader()
    │               │       ├─ online: WallDataset
    │               │       │   └─> make_dataloader_for_prebatched_ds()
    │               │       ├─> _create_wall_probing_datasets()
    │               │       │   ├─> probe_ds (WallDataset)
    │               │       │   ├─> probe_val_ds (WallDataset)
    │               │       │   └─> extra_datasets
    │               │       │       ├─ WallPassingTestDataset
    │               │       │       └─ BorderPassingTestDataset
    │               │       └─> Datasets(ds, probing_datasets)
    │               │
    │               ├─ DatasetType.WallExpert
    │               │   └─> _create_wall_expert_datasets()
    │               │       └─> WrappedWallExpertDataset
    │               │
    │               ├─ DatasetType.D4RL
    │               │   └─> _create_d4rl_datasets()
    │               │       ├─> D4RLDataset (訓練)
    │               │       │   ├─ _prepare_ds()
    │               │       │   ├─ _prepare_saved_ds()
    │               │       │   └─ _prepare_mixed_ds()
    │               │       ├─> D4RLDataset (プロービング訓練)
    │               │       ├─> D4RLDataset (プロービング検証)
    │               │       └─> Datasets(ds, probing_datasets)
    │               │
    │               └─ DatasetType.LocoMaze
    │                   └─> _create_locomaze_datasets()
    │
    └─> make_dataloader() or make_dataloader_for_prebatched_ds()
            ├─> torch.utils.data.DataLoader (if needed)
            ├─> Normalizer.build_normalizer()
            │   ├─ データセットから100サンプル取得
            │   ├─ 各フィールドの平均・標準偏差を計算
            │   └─ Normalizerインスタンス作成
            └─> NormalizedDataLoader
                └─ イテレーション時に正規化を適用
```

---

## 7. データセット出力の構造

### 7.1 共通フィールド
すべてのデータセットは以下の基本フィールドを含む`NamedTuple`を返します:

```python
states: torch.Tensor      # 状態（画像または固有受容情報）
actions: torch.Tensor     # アクション
```

### 7.2 データセット固有フィールド

**DotDataset / WallDataset**:
```python
locations: torch.Tensor   # エージェントの位置座標
bias_angle: torch.Tensor  # アクションバイアス角度
wall_x: torch.Tensor      # 壁のx座標 (Wallのみ)
door_y: torch.Tensor      # ドアのy座標 (Wallのみ)
```

**D4RLDataset**:
```python
propio_pos: torch.Tensor (optional)  # 固有受容位置
propio_vel: torch.Tensor (optional)  # 固有受容速度
goal: torch.Tensor (optional)        # ゴール情報
```

### 7.3 テンソルの形状

**訓練時** (バッチ化後):
- `states`: `[Batch, Time, Channels, Height, Width]` または `[Batch, Time, Dim]`
- `actions`: `[Batch, Time-1, Action_Dim]`
- `locations`: `[Batch, Time, 2]` (2D環境の場合)

**注意**: 訓練ループで`transpose(0, 1)`が呼ばれるため、実際のモデル入力は`[Time, Batch, ...]`になります ([train.py:361-362](pldm/train.py#L361-L362))。

---

## 8. 主要な依存ファイル

### データセット関連
- [pldm/data/dataset_factory.py](pldm/data/dataset_factory.py) - データセットファクトリ
- [pldm/data/enums.py](pldm/data/enums.py) - データ設定と列挙型
- [pldm/data/utils.py](pldm/data/utils.py) - DataLoader作成とユーティリティ

### 環境データセット実装
- [pldm_envs/wall/data/single.py](pldm_envs/wall/data/single.py) - Dotデータセット
- [pldm_envs/wall/data/wall.py](pldm_envs/wall/data/wall.py) - Wallデータセット
- [pldm_envs/wall/data/offline_wall.py](pldm_envs/wall/data/offline_wall.py) - オフラインWallデータセット
- [pldm_envs/wall/data/wall_expert.py](pldm_envs/wall/data/wall_expert.py) - エキスパートデータセット
- [pldm_envs/diverse_maze/d4rl.py](pldm_envs/diverse_maze/d4rl.py) - D4RLデータセット

### ユーティリティ
- [pldm_envs/utils/normalizer.py](pldm_envs/utils/normalizer.py) - データ正規化
- [pldm_envs/wall/data/wall_utils.py](pldm_envs/wall/data/wall_utils.py) - Wall環境ユーティリティ

---

## 9. 重要なポイント

### 9.1 正規化の重要性
- ほとんどの場合、`normalize=True`が推奨される
- 正規化は学習の安定性と収束速度に大きく影響
- 訓練データから計算した統計量を検証データにも適用

### 9.2 メモリ管理
- **オフラインデータセット**: `lazy_load=True`でメモリ使用量を削減
- **D4RL画像データ**: mmapモードで大容量データを効率的に扱う
- **num_workers**: マルチプロセスでデータロードを高速化（GPUトレーニング時）

### 9.3 プロービングデータセット
- 表現学習の品質を評価するための別データセット
- 訓練データと異なる設定（例: `n_steps`が異なる）で作成可能
- 線形プローブや評価タスクで使用

### 9.4 データセットのバッチ化
- **PyTorchスタイル**: `__getitem__`でサンプルを返し、DataLoaderがバッチ化
- **プリバッチスタイル**: データセット自体がバッチを返す（WallDatasetなど）
