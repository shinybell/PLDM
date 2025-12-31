# Custom Map Environment Guide

2次元配列からカスタムマップを作成できる `CustomMapEnv` の使い方ガイドです。

## 基本的な使い方

### 1. シンプルな例

```python
import gymnasium as gym
import pldm_envs.minigrid

# カスタムマップを定義 (1=壁, 0=通路)
my_map = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]

# 環境を作成
env = gym.make('MiniGrid-CustomMap-v0', map_array=my_map)

# 環境を実行
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### 2. エージェントとゴールの位置を指定

```python
# カスタムマップ
my_map = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

# エージェントとゴールの位置を指定
env = gym.make(
    'MiniGrid-CustomMap-v0',
    map_array=my_map,
    agent_start_pos=(1, 1),  # (x, y) - 左上の通路
    goal_pos=(5, 5),         # (x, y) - 右下の通路
    max_steps=100
)
```

### 3. データ生成スクリプトと組み合わせる

```bash
# generate_data.pyを使用
python pldm_envs/minigrid/data_generation/generate_data.py \
    --env_name MiniGrid-CustomMap-v0 \
    --n_episodes 1000 \
    --max_steps 256 \
    --resize 64 \
    --output_path data/minigrid/custom_map_data.npz
```

**注意**: コマンドラインからはカスタムマップを渡せないため、デフォルトマップが使用されます。カスタムマップを使う場合は、Pythonスクリプトで直接環境を作成してください。

## マップの仕様

### 必須条件

1. **2次元配列**: `List[List[int]]` または numpy array
2. **値**: `1` = 壁、`0` = 通路
3. **外周は壁**: すべての外周セルは `1` でなければなりません

```python
# ✅ 正しい例
correct_map = [
    [1, 1, 1, 1, 1],  # すべて壁
    [1, 0, 0, 0, 1],  # 両端が壁
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]   # すべて壁
]

# ❌ 間違った例（外周に通路がある）
wrong_map = [
    [1, 1, 1, 1, 0],  # 右端が通路 → エラー
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
```

### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `map_array` | List[List[int]] | 5x5デフォルトマップ | 1=壁、0=通路の2次元配列 |
| `agent_start_pos` | tuple (x, y) | 自動配置 | エージェントの開始位置 |
| `agent_start_dir` | int | 0 | エージェントの初期方向 (0-3) |
| `goal_pos` | tuple (x, y) | 右下の通路 | ゴールの位置 |
| `max_steps` | int | 256 | 最大ステップ数 |

### 座標系

- **(x, y)** 形式: x=列（左から右）、y=行（上から下）
- 原点 (0, 0) は左上
- 配列のインデックスは `map_array[y][x]`

```python
# 例: 5x5マップ
#    x: 0  1  2  3  4
# y
# 0    1  1  1  1  1
# 1    1  0  0  0  1
# 2    1  1  1  0  1
# 3    1  0  0  0  1
# 4    1  1  1  1  1

# (1, 1) は左上の通路
# (3, 3) は右下の通路
```

## マップ例

### 1. シンプルな廊下

```python
simple_corridor = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
```

### 2. 十字路

```python
cross_roads = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]
]
```

### 3. 迷路

```python
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
```

### 4. 部屋と中央の障害物

```python
room_with_obstacle = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]
]
```

## テストスクリプト

環境が正しく動作するかテスト:

```bash
# テストスクリプトを実行
source .venv/bin/activate
PYTHONPATH=/Users/shunsei/works/MatsuoLab/WorldModel/PLDM:$PYTHONPATH \
python pldm_envs/minigrid/test_custom_map.py
```

## Pythonスクリプトでのデータ生成

カスタムマップでデータを生成する場合は、直接Pythonスクリプトを書く必要があります:

```python
import numpy as np
import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
import pldm_envs.minigrid

# カスタムマップ
my_map = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

# 環境作成
env = gym.make('MiniGrid-CustomMap-v0', map_array=my_map, max_steps=256)
env = RGBImgObsWrapper(env, tile_size=8)
env = ImgObsWrapper(env)

# データ収集
episodes = []
for i in range(100):
    obs_list = []
    action_list = []

    obs, info = env.reset(seed=i)
    obs_list.append(obs)
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        obs_list.append(obs)
        action_list.append(action)
        done = terminated or truncated

    episodes.append({
        'observations': np.array(obs_list),
        'actions': np.array(action_list)
    })

# データ保存
np.savez_compressed(
    'custom_map_data.npz',
    observations=np.array([ep['observations'] for ep in episodes], dtype=object),
    actions=np.array([ep['actions'] for ep in episodes], dtype=object)
)
```

## エラー処理

### よくあるエラー

1. **外周が壁でない**
```python
# エラー: ValueError: All outer cells of map_array must be walls (1)
bad_map = [
    [1, 1, 1, 0, 1],  # 上の外周に通路
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]
```

2. **エージェント/ゴールの位置が壁**
```python
my_map = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]

# エラー: ValueError: Agent start position (2, 2) is a wall in the map
env = gym.make('MiniGrid-CustomMap-v0',
               map_array=my_map,
               agent_start_pos=(2, 2))  # この位置は壁
```

3. **2次元配列でない**
```python
# エラー: ValueError: map_array must be a 2D array
bad_map = [1, 0, 1, 0, 1]  # 1次元配列
```

## 応用例

### 複数の複雑さレベルのマップを作成

```python
# Level 1: Simple
level1_map = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

# Level 2: Medium
level2_map = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

# Level 3: Complex
level3_map = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

# それぞれの環境を作成
env1 = gym.make('MiniGrid-CustomMap-v0', map_array=level1_map)
env2 = gym.make('MiniGrid-CustomMap-v0', map_array=level2_map)
env3 = gym.make('MiniGrid-CustomMap-v0', map_array=level3_map)
```

## 関連ファイル

- [envs/long_horizon_envs.py](envs/long_horizon_envs.py#L327) - CustomMapEnvの実装
- [test_custom_map.py](test_custom_map.py) - テストスクリプト
- [__init__.py](__init__.py#L22) - 環境の登録

## まとめ

`CustomMapEnv` を使えば:
- ✅ 2次元配列から簡単にカスタムマップを作成
- ✅ エージェントとゴールの位置を自由に設定
- ✅ 任意の複雑さの迷路を実験可能
- ✅ 既存のツール（データ生成、可視化）と互換性あり

研究で特定のマップ構造を試したい場合に最適です！
