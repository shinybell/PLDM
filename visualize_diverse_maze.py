"""
DiverseMazeデータセットの観測画像を可視化するスクリプト
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 画像データのパスを確認
images_path = Path("/pldm_envs/diverse_maze/presaved_datasets/5maps/images.npy")

# ローカルパスの可能性もあるので、複数試す
possible_paths = [
    "/pldm_envs/diverse_maze/presaved_datasets/5maps/images.npy",
    "pldm_envs/diverse_maze/presaved_datasets/5maps/images.npy",
    "/Users/shunsei/works/MatsuoLab/WorldModel/PLDM/pldm_envs/diverse_maze/presaved_datasets/5maps/images.npy",
]

images = None
used_path = None

for path_str in possible_paths:
    path = Path(path_str)
    if path.exists():
        print(f"Found images at: {path}")
        images = np.load(path, mmap_mode='r')
        used_path = path
        break

if images is None:
    print("画像ファイルが見つかりません。以下のパスを確認してください:")
    for p in possible_paths:
        print(f"  - {p}")
    print("\n利用可能なデータセットディレクトリを探します...")

    # pldm_envsディレクトリを探す
    import os
    for root, dirs, files in os.walk("pldm_envs/diverse_maze", topdown=True):
        if "images.npy" in files:
            print(f"Found: {os.path.join(root, 'images.npy')}")
else:
    print(f"Images shape: {images.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Images range: [{images.min()}, {images.max()}]")

    # ランダムに5枚サンプリング
    n_samples = min(5, len(images))
    indices = np.random.choice(len(images), n_samples, replace=False)

    # 可視化
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    if n_samples == 1:
        axes = [axes]

    for idx, ax in zip(indices, axes):
        img = images[idx]

        # 画像の形状を確認
        print(f"Sample {idx} shape: {img.shape}, dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")

        # チャンネルの順序を確認して調整
        if img.shape[0] == 3:  # (3, H, W) の場合
            img = np.transpose(img, (1, 2, 0))  # (H, W, 3) に変換

        # 正規化されている場合は0-255にスケール
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        ax.imshow(img)
        ax.set_title(f"Sample {idx}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("diverse_maze_samples.png", dpi=150, bbox_inches='tight')
    print(f"\n可視化結果を 'diverse_maze_samples.png' に保存しました")
    plt.close()

    # 1枚目を詳細に表示
    print(f"\n最初のサンプル（index {indices[0]}）の詳細:")
    first_img = images[indices[0]]
    if first_img.shape[0] == 3:
        first_img = np.transpose(first_img, (1, 2, 0))
    if first_img.max() <= 1.0:
        first_img = (first_img * 255).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(first_img)
    plt.title(f"DiverseMaze Observation (index {indices[0]})")
    plt.axis('off')
    plt.savefig("diverse_maze_single.png", dpi=150, bbox_inches='tight')
    print("詳細画像を 'diverse_maze_single.png' に保存しました")
    plt.close()
