FROM python:3.10-slim

# システム依存関係のインストール
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1-mesa-dev \
    libgl1 \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    libglfw3 \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# MuJoCo 2.1.0のインストール
RUN mkdir -p /root/.mujoco && \
    cd /root/.mujoco && \
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xf mujoco210-linux-x86_64.tar.gz && \
    rm mujoco210-linux-x86_64.tar.gz

# 環境変数の設定
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV MUJOCO_GL=egl

# 作業ディレクトリ
WORKDIR /workspace

# Pythonパッケージの依存関係をコピー
COPY requirements.txt .

# 依存関係をインストール（Cythonを先に古いバージョンで固定）
RUN pip install 'Cython<3.0.0' && \
    pip install torch==2.5.1 torchvision==0.20.1 gymnasium==1.0.0 gym==0.23.1 numpy==1.26.4 mujoco==3.2.6 && \
    pip install mujoco-py==2.1.2.14 --no-build-isolation && \
    pip install d4rl==1.1 wandb tqdm zarr arm-pytorch-utilities

# プロジェクト全体をコピー
COPY . .

# パッケージをインストール
RUN pip install -e .

# デフォルトコマンド
CMD ["/bin/bash"]
