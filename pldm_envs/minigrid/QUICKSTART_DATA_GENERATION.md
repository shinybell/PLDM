# MiniGrid データ生成 クイックスタート

## 概要

MiniGrid LongHorizon環境（Level 1-3）でPLDM学習用のオフラインデータを生成する方法を説明します。

## データ生成の流れ

```
環境作成 → データ収集 → 保存 → PLDM学習
```

## 1. テスト実行（まず試す）

少量のデータで動作確認：

```bash
# 仮想環境をアクティベート
source .venv/bin/activate

# Level 1で10エピソード生成（72x72）
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 10 \
  --obs_size 72 \
  --output_path data/minigrid/test_level1.npz
```

## 2. 訓練データ生成（本番）

### オプション A: 全レベル一括生成（推奨）

```bash
# 64x64で各レベル1000エピソード
bash pldm_envs/minigrid/data_generation/generate_all_levels.sh \
  --n_episodes 1000 \
  --obs_size 64

# 72x72で各レベル5000エピソード
bash pldm_envs/minigrid/data_generation/generate_all_levels.sh \
  --n_episodes 5000 \
  --obs_size 72
```

生成されるファイル:
```
data/minigrid/
├── level1_64x64_train.npz  (または 72x72)
├── level2_64x64_train.npz
└── level3_64x64_train.npz
```

### オプション B: 個別レベル生成

```bash
# Level 1のみ（64x64）
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 1000 \
  --obs_size 64 \
  --output_path data/minigrid/level1_train.npz

# Level 2（72x72）
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level2-v0 \
  --n_episodes 1000 \
  --obs_size 72 \
  --output_path data/minigrid/level2_72x72_train.npz
```

## 3. 評価データ生成

```bash
# 評価用（100エピソード、異なるseed）
bash pldm_envs/minigrid/data_generation/generate_all_levels.sh \
  --n_episodes 100 \
  --obs_size 64 \
  --seed 12345 \
  --data_dir data/minigrid/val
```

## 4. データ確認

生成したデータの内容を確認：

```bash
python -c "
import numpy as np

# データ読み込み
data = np.load('data/minigrid/level1_64x64_train.npz', allow_pickle=True)

print('データセット情報:')
print(f'  エピソード数: {len(data[\"observations\"])}')
print(f'  観測形状: {data[\"observations\"][0].shape}')
print(f'  観測型: {data[\"observations\"][0].dtype}')
print(f'  値範囲: [{data[\"observations\"][0].min()}, {data[\"observations\"][0].max()}]')
print(f'  アクション数: {len(data[\"actions\"][0])}')
"
```

## 5. 可視化（オプション）

生成したデータをサンプル表示：

```python
import numpy as np
import matplotlib.pyplot as plt

# データ読み込み
data = np.load('data/minigrid/level1_64x64_train.npz', allow_pickle=True)

# 最初のエピソードの最初の4フレームを表示
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    axes[i].imshow(data['observations'][0][i])
    axes[i].set_title(f'Frame {i}')
    axes[i].axis('off')
plt.savefig('sample_episode.png')
print('Saved: sample_episode.png')
```

## データフォーマット

### 生成されるデータ

```python
{
    'observations': (N, T, H, W, C)  # N=エピソード数, T=ステップ数
    'actions': (N, T-1)               # 離散アクション（0-6）
    'rewards': (N, T-1)               # 報酬（通常0, ゴール時1）
    'dones': (N, T-1)                 # 終了フラグ
}
```

### 観測サイズの選択

| サイズ | 用途 | メモリ使用量 | 計算速度 |
|-------|------|------------|---------|
| 64x64 | 標準（TwoRoomsと同じ） | 中 | 速い |
| 72x72 | 高解像度 | 大 | 中 |
| 84x84 | より高解像度 | 最大 | 遅い |

**推奨**: まず64x64で実験し、必要に応じて72x72に変更

## よくある質問

### Q1: どのくらいのエピソード数が必要？

- **試験的**: 100-500エピソード
- **標準**: 1000-2000エピソード（TwoRooms相当）
- **本格的**: 5000-10000エピソード

### Q2: PyTorch形式（CHW）は必要？

デフォルトのHWC形式で保存し、学習時にデータローダーで変換する方が柔軟です。
ただし、すべてCHW形式にしたい場合：

```bash
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 1000 \
  --obs_size 64 \
  --channel_first \
  --normalize \
  --output_path data/minigrid/level1_pytorch.npz
```

### Q3: メモリ不足エラーが出る

大量データ生成時は分割生成：

```bash
# 1000エピソードずつ10回生成
for i in {0..9}; do
  PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
    --env_name MiniGrid-LongHorizon-Level1-v0 \
    --n_episodes 1000 \
    --obs_size 64 \
    --seed $((42 + i * 1000)) \
    --output_path data/minigrid/level1_batch_${i}.npz
done
```

### Q4: ランダムポリシー以外は使える？

現在はランダムポリシーのみ実装されています。
学習済みポリシーを使いたい場合は`generate_pldm_data.py`を拡張してください。

## 次のステップ：PLDM学習

データ生成後の流れ：

1. **データセットクラス作成**
   - `pldm_envs/wall`や`pldm_envs/diverse_maze`のデータローダーを参考
   - MiniGrid用のDatasetクラスを実装

2. **設定ファイル作成**
   - 学習ハイパーパラメータ
   - モデルアーキテクチャ
   - データパス

3. **学習実行**
   ```bash
   python train.py --config configs/minigrid_level1.yaml
   ```

4. **評価・可視化**
   - 予測精度の評価
   - 生成画像の可視化
   - レベル間の比較

詳細は[data_generation/README.md](data_generation/README.md)を参照してください。

## トラブルシューティング

### 環境が見つからない

```
gym.error.UnregisteredEnv: Environment MiniGrid-LongHorizon-Level1-v0 doesn't exist
```

→ `import pldm_envs.minigrid`が実行されているか確認

### PYTHONPATH エラー

```
ModuleNotFoundError: No module named 'pldm_envs'
```

→ `PYTHONPATH=$PWD:$PYTHONPATH`を付けて実行

### PIL/cv2 エラー

wrappers.pyはPILを使用（cv2不要）。エラーが出る場合：

```bash
pip install Pillow
```

## まとめ

```bash
# 1. テスト（10エピソード）
PYTHONPATH=$PWD:$PYTHONPATH python pldm_envs/minigrid/data_generation/generate_pldm_data.py \
  --env_name MiniGrid-LongHorizon-Level1-v0 \
  --n_episodes 10 \
  --obs_size 72 \
  --output_path data/minigrid/test.npz

# 2. 本番（全レベル1000エピソード）
bash pldm_envs/minigrid/data_generation/generate_all_levels.sh \
  --n_episodes 1000 \
  --obs_size 72

# 3. データ確認
ls -lh data/minigrid/*.npz

# 4. PLDM学習へ進む
```
