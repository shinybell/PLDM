# Docker環境でPLDMを実行する

Apple Silicon (M1/M2/M3) Macなど、mujoco-pyの互換性問題がある環境向けのDockerセットアップガイドです。

## 前提条件

- Docker Desktop for Mac がインストールされていること
- （オプション）docker-compose がインストールされていること

## クイックスタート

### 方法1: docker-composeを使う（推奨）

```bash
# 1. Dockerイメージをビルド
docker-compose build

# 2. コンテナを起動してシェルに入る
docker-compose run --rm pldm

# 3. コンテナ内でサンプルデータを生成
bash pldm_envs/diverse_maze/data_generation/generate_sample_data.sh
```

### 方法2: dockerコマンドを直接使う

```bash
# 1. Dockerイメージをビルド
docker build -t pldm:latest .

# 2. コンテナを起動してシェルに入る
docker run -it --rm -v $(pwd):/workspace pldm:latest

# 3. コンテナ内でサンプルデータを生成
bash pldm_envs/diverse_maze/data_generation/generate_sample_data.sh
```

## 詳細な使い方

### イメージのビルド

初回、またはDockerfileやrequirements.txtを変更した場合：

```bash
docker-compose build
# または
docker build -t pldm:latest .
```

### コンテナの起動

```bash
# docker-composeの場合
docker-compose run --rm pldm bash

# dockerコマンドの場合
docker run -it --rm \
  -v $(pwd):/workspace \
  -e MUJOCO_GL=egl \
  pldm:latest bash
```

### データ生成

コンテナ内で：

```bash
# サンプルデータ（10エピソード）
bash pldm_envs/diverse_maze/data_generation/generate_sample_data.sh

# フルデータセット（元の論文データ）
bash pldm_envs/diverse_maze/data_generation/generate_all_datasets_og.sh
```

### トレーニング

```bash
# コンテナ内で
python pldm/train.py --config pldm/configs/diverse_maze/icml/small_diverse_5maps.yaml
```

## データの永続化

生成されたデータは、以下のいずれかの方法で永続化できます：

### 方法1: ボリュームマウント（docker-compose）

`docker-compose.yml`のvolumesセクションで設定済み：
- `.:/workspace` - プロジェクト全体をマウント
- `pldm-data:/workspace/pldm_envs/diverse_maze/data` - データ専用ボリューム

データはホストマシンの`pldm_envs/diverse_maze/data/`に保存されます。

### 方法2: ホストにコピー

```bash
# コンテナからホストへ
docker cp <container_id>:/workspace/pldm_envs/diverse_maze/data ./pldm_envs/diverse_maze/
```

## トラブルシューティング

### ビルドが遅い

```bash
# ビルドキャッシュを使わずに再ビルド
docker-compose build --no-cache
```

### MuJoCo関連のエラー

コンテナ内で確認：

```bash
python -c "import mujoco_py; print('Success!')"
```

### メモリ不足

Docker Desktopの設定で、メモリを増やす：
1. Docker Desktop > Settings > Resources
2. Memory を 8GB以上に設定

### 権限エラー

```bash
# ボリュームの権限を修正
docker-compose run --rm pldm chown -R $(id -u):$(id -g) /workspace
```

## GPU対応（オプション）

NVIDIA GPUを使用する場合：

```yaml
# docker-compose.ymlに追加
services:
  pldm:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## コンテナのクリーンアップ

```bash
# 停止したコンテナを削除
docker-compose down

# イメージも削除
docker-compose down --rmi all

# ボリュームも削除（データが消えます！）
docker-compose down -v
```

## ヒント

1. **開発中**: コンテナを起動したままにして、別のターミナルから接続
   ```bash
   # ターミナル1
   docker-compose run --rm pldm bash

   # ターミナル2（別のシェル）
   docker exec -it pldm-dev bash
   ```

2. **ワンライナー実行**: コンテナに入らずにコマンド実行
   ```bash
   docker-compose run --rm pldm bash pldm_envs/diverse_maze/data_generation/generate_sample_data.sh
   ```

3. **WandBの使用**: APIキーを環境変数で渡す
   ```bash
   docker-compose run --rm -e WANDB_API_KEY=<your-key> pldm
   ```
