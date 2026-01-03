"""
複数のワーカーで生成したデータセットを結合するスクリプト

使用方法:
    # 4つのワーカーで生成したデータを結合 (ディレクトリ形式で保存)
    python pldm_envs/minigrid/data_generation/merge_datasets.py \
        --input_pattern "data/minigrid/worker_*.npz" \
        --output_path data/minigrid/train

    # .npz形式で保存 (メモリマップを使用してメモリ節約)
    python pldm_envs/minigrid/data_generation/merge_datasets.py \
        --input_pattern "data/minigrid/worker_*.npz" \
        --output_path data/minigrid/train.npz
"""

import numpy as np
import argparse
from pathlib import Path
import glob
import shutil
import tempfile
import os
from tqdm import tqdm


def get_dataset_info(input_files):
    """
    入力ファイルからデータセットの総サイズと形状を取得
    """
    total_episodes = 0
    shapes = {}
    dtypes = {}
    keys = []

    print("Analyzing input files...")
    for i, file_path in enumerate(tqdm(input_files, desc="Scanning files")):
        data = np.load(file_path, allow_pickle=True)
        
        if i == 0:
            keys = list(data.keys())
            for k in keys:
                shapes[k] = data[k].shape[1:]  # (N, ...) -> (...)
                dtypes[k] = data[k].dtype
        
        # Check consistency
        # for k in keys:
        #     assert data[k].shape[1:] == shapes[k], f"Shape mismatch for {k} in {file_path}"
        
        total_episodes += data[keys[0]].shape[0]

    return total_episodes, shapes, dtypes, keys


def merge_datasets_memmap(input_files, output_path, is_dir=False):
    """
    メモリマップを使用してデータセットを結合
    """
    total_episodes, shapes, dtypes, keys = get_dataset_info(input_files)
    
    print(f"\nTotal episodes: {total_episodes}")
    print(f"Keys: {keys}")
    
    # 出力先がディレクトリの場合は直接そこに保存
    # .npzの場合は一時ディレクトリに保存してから圧縮
    if is_dir:
        work_dir = Path(output_path)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp()
        work_dir = Path(temp_dir)
    
    try:
        # メモリマップファイルの作成
        memmaps = {}
        for k in keys:
            shape = (total_episodes,) + shapes[k]
            filename = work_dir / f"{k}.npy"
            print(f"Allocating {filename} with shape {shape} ({dtypes[k]})")
            
            # ディスク上に領域を確保
            memmaps[k] = np.lib.format.open_memmap(
                filename, mode='w+', dtype=dtypes[k], shape=shape
            )
        
        # データのコピー
        current_idx = 0
        print("\nCopying data...")
        for file_path in tqdm(input_files, desc="Merging"):
            data = np.load(file_path, allow_pickle=True)
            n_episodes = data[keys[0]].shape[0]
            
            for k in keys:
                memmaps[k][current_idx : current_idx + n_episodes] = data[k]
                # メモリ解放のためにflush（必須ではないが念のため）
                # memmaps[k].flush()
            
            current_idx += n_episodes
            
            # メモリ解放
            del data
            
        # 全てのデータをディスクに書き込み
        for k in keys:
            del memmaps[k]  # これでファイルが閉じられる
            
        # .npzの場合は圧縮して保存
        if not is_dir:
            print(f"\nCompressing to {output_path}...")
            # 一時ファイルのパスリスト
            npy_files = {k: work_dir / f"{k}.npy" for k in keys}
            
            # np.savez_compressedはファイルパスを受け取れないため、
            # 一度読み込む必要があるが、mmap_mode='r'で読み込めばメモリを圧迫しない
            data_dict = {
                k: np.load(npy_files[k], mmap_mode='r') for k in keys
            }
            
            np.savez_compressed(output_path, **data_dict)
            
    finally:
        # 一時ディレクトリの削除（.npzの場合のみ）
        if not is_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="複数のデータセットを結合 (メモリ効率版)")
    parser.add_argument(
        "--input_pattern",
        type=str,
        default=None,
        help="入力ファイルのパターン (例: 'data/worker_*.npz')",
    )
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        default=None,
        help="入力ファイルのリスト",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="出力パス (.npz または ディレクトリ)"
    )

    args = parser.parse_args()

    # 入力ファイルを取得
    if args.input_pattern is not None:
        input_files = sorted(glob.glob(args.input_pattern))
        if len(input_files) == 0:
            raise ValueError(f"No files found matching pattern: {args.input_pattern}")
    elif args.input_files is not None:
        input_files = sorted(args.input_files)
    else:
        raise ValueError("Either --input_pattern or --input_files must be specified")

    print("=" * 60)
    print("Dataset Merging (Memory Efficient)")
    print("=" * 60)
    print(f"Input files ({len(input_files)}):")
    # for f in input_files[:5]:
    #     print(f"  - {f}")
    # if len(input_files) > 5:
    #     print(f"  ... and {len(input_files) - 5} more")
    print(f"Output: {args.output_path}")
    
    is_dir = not args.output_path.endswith('.npz')
    if is_dir:
        print("Output format: Directory of .npy files (Recommended for training)")
    else:
        print("Output format: Compressed .npz file")
    print("=" * 60)

    # データセットを結合
    merge_datasets_memmap(input_files, args.output_path, is_dir=is_dir)

    print("\n" + "=" * 60)
    print("Dataset merging completed!")
    print("=" * 60)
    
    # ファイルサイズ
    output_path = Path(args.output_path)
    if is_dir:
        total_size = sum(f.stat().st_size for f in output_path.glob('**/*') if f.is_file())
        print(f"Total size: {total_size / (1024 * 1024):.1f} MB")
    else:
        print(f"File size: {output_path.stat().st_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
