"""
MiniGridの画像データをnumpy配列に変換するスクリプト

Diverse Mazeのpostprocess_images.pyを参考にMiniGrid用に実装。

使用方法:
    python pldm_envs/minigrid/data_generation/postprocess_images.py \\
        --data_path pldm_envs/minigrid/data/sample_10episodes
"""

import concurrent.futures
import collections
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from functools import partial
import sys
import argparse
import zarr
import torch

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    import warnings
    warnings.warn("opencv-python not found. Resizing will use PIL instead.")


def resize_image_cv2(image, target_size):
    """OpenCVでリサイズ"""
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)


def resize_image_pil(image, target_size):
    """PILでリサイズ"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize((target_size, target_size), Image.LANCZOS)
    return np.array(image)


def process_image(image_path: str, target_size: int):
    """
    画像を読み込み、リサイズして返す

    Args:
        image_path: 画像ファイルパス
        target_size: リサイズ後のサイズ

    Returns:
        リサイズされた画像 [target_size, target_size, 3]
    """
    try:
        with Image.open(image_path) as image:
            image_array = np.array(image)

            # リサイズ
            if HAS_CV2:
                resized = resize_image_cv2(image_array, target_size)
            else:
                resized = resize_image_pil(image, target_size)

            # RGB形式を確認
            if len(resized.shape) == 2:
                # グレースケールの場合はRGBに変換
                resized = np.stack([resized] * 3, axis=-1)

            return resized
    except Exception as e:
        episode_idx = int(image_path.split("/")[-1].split("_")[0])
        print(f"Error processing {image_path}: {e}")
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)


def list_images(args, config: dict):
    """
    処理する画像のリストを作成

    Args:
        args: コマンドライン引数
        config: メタデータ設定

    Returns:
        画像ファイル名のリスト
    """
    image_list = []
    episode_length = config["episode_length"]
    n_episodes = config["n_episodes"]

    for i in range(n_episodes):
        if i >= args.stop_at_episode:
            break

        # episode_length + 1: 初期状態 + episode_lengthステップ
        for j in range(episode_length + 1):
            image_list.append(f"{i}_{j}.png")

    return image_list


def main():
    parser = argparse.ArgumentParser(description="MiniGrid画像をnumpy配列に保存")
    parser.add_argument("--data_path", type=str, required=True, help="データパス")
    parser.add_argument("--quick_debug", action="store_true", help="デバッグモード")
    parser.add_argument("--num_workers", type=int, default=10, help="並列処理ワーカー数")
    parser.add_argument("--stop_at_episode", type=int, default=sys.maxsize, help="処理を停止するエピソード番号")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    config_path = data_path / "metadata.pt"

    if not config_path.exists():
        print(f"Error: metadata.pt not found at {config_path}")
        return

    config = torch.load(config_path)

    # 画像リストを作成
    image_list = list_images(args, config=config)
    num_images = len(image_list)

    print(f"Total images to process: {num_images}")

    # 画像をエピソード別に整理
    input_image_path = data_path / "images"

    if not input_image_path.exists():
        print(f"Error: images directory not found at {input_image_path}")
        return

    episode_dict = collections.OrderedDict()
    for image_file in sorted(image_list):
        image_path = input_image_path / image_file
        if not image_path.exists():
            print(f"Warning: {image_path} does not exist, skipping")
            continue

        episode_idx, timestep = image_file.split("_")
        episode_idx = int(episode_idx)
        timestep = int(timestep[:-4])  # ".png"を除去

        if episode_idx not in episode_dict:
            episode_dict[episode_idx] = []
        episode_dict[episode_idx].append((timestep, str(image_path)))

    # 画像サイズを取得
    img_size = config["img_size"]
    target_size = img_size[0]  # [H, W, C] -> H (正方形を想定)

    print(f"Target image size: {target_size}x{target_size}")

    # 実際に存在する画像数を数える
    actual_num_images = sum(len(imgs) for imgs in episode_dict.values())
    data_shape = (actual_num_images, target_size, target_size, 3)
    data_chunks = (min(1000, actual_num_images), target_size, target_size, 3)

    print(f"Actual images found: {actual_num_images}")
    print(f"Data shape: {data_shape}")
    print(f"Chunk size: {data_chunks}")

    # Zarrデータセットを初期化
    zarr_path = data_path / "images.zarr"
    zarr_dataset = zarr.open(
        str(zarr_path),
        mode='w',
        shape=data_shape,
        chunks=data_chunks,
        dtype=np.uint8,
    )

    # 画像処理関数
    process_image_with_size = partial(process_image, target_size=target_size)

    ctr = 0

    print(f"\nLoading images from {input_image_path}")

    # 各エピソードの画像を読み込んで処理
    for i, (episode_idx, image_list) in enumerate(tqdm(
        sorted(episode_dict.items()),
        desc="Processing episodes",
        total=len(episode_dict),
    )):
        # タイムステップでソート
        image_list = sorted(image_list, key=lambda x: x[0])

        # 並列処理で画像を読み込み
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            images = list(
                executor.map(
                    process_image_with_size,
                    [image_path for _, image_path in image_list],
                )
            )

        # Zarrに保存
        if len(images) > 0:
            zarr_dataset[ctr : ctr + len(images)] = np.stack(images)
            ctr += len(images)

        if args.quick_debug and i > 2:
            break

    # Zarrからnumpy配列に変換して保存
    print(f"\nConverting zarr to numpy array...")
    npy_path = data_path / "images.npy"
    np.save(str(npy_path), zarr_dataset[:])

    print(f"\nCompleted!")
    print(f"  Zarr saved to: {zarr_path}")
    print(f"  Numpy array saved to: {npy_path}")
    print(f"  Total images processed: {ctr}")
    print(f"  Final shape: {zarr_dataset.shape}")

    # ファイルサイズ
    if npy_path.exists():
        file_size_mb = npy_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    from pathlib import Path
    main()
