import argparse
import yaml

import numpy as np
import torch
from tqdm import tqdm

from pldm_envs.wall.data.wall import WallDataset, WallDatasetConfig
from pldm_envs.wall.save_wall_ds import update_config_from_yaml


def parse_args():
    """A function to parse arguments with argparse:
    - data_paths: a list of paths to the data files
    - wc_rate: target wall crossing rate in the new dataset
    - output_path: path to save the new dataset
    """

    parser = argparse.ArgumentParser(
        description="Render images in a dataset without image observations",
    )
    parser.add_argument("--input_path", type=str, help="Path to the data file")
    parser.add_argument("--config", type=str, help="Path to the dataset config file")
    parser.add_argument(
        "--output_path",
        type=str,
        default="new_dataset.npz",
        help="Path to save the new dataset with images rendered.",
    )
    parser.add_argument(
        "--render_batch_size",
        type=int,
        default=1000,
        help="Number of trajectories render at a time",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    # データをメモリーマップで読み込む (メモリ効率の維持)
    # data_mmap は辞書のように振る舞いますが、巨大な配列は物理メモリにロードされません。
    data_mmap = np.load(args.input_path, mmap_mode="r") 
    locations = data_mmap["locations"] 
    num_locations = len(locations)

    print(f"✅ データセットを読み込みました。総ロケーション数: {num_locations}")

    # 1. Configの読み込みとWallDatasetの初期化
    try:
        with open(args.config, "r") as file:
            yaml_config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"❌ 設定ファイルが見つかりません: {args.config}")
        return
        
    config = update_config_from_yaml(WallDatasetConfig, yaml_config)
    ds = WallDataset(config)
    H, W = config.img_size, config.img_size 

    # 2. Wall情報をレンダリングして保存データに格納
    wall_info = ds.sample_walls()
    walls_tensor = ds.render_walls(*wall_info)
    # 最初の壁のみを使用し、NumPy配列 (H, W, 1) に変換
    walls_numpy = walls_tensor[0].unsqueeze(-1).numpy() 
    print(f"🖼️ Wall情報がレンダリングされました。形状: {walls_numpy.shape}")

    # 3. データを格納する最終的な辞書を初期化
    final_data = {}
    
    # locations以外のキーをコピー (通常はサイズが小さいメタデータ)
    for key, value in data_mmap.items():
        if key != "locations":
            final_data[key] = value.copy() 
    
    # locationsも最終データに追加 (locations自体は大量だが、ここではコピーを許容)
    final_data["locations"] = locations.copy()
    
    # 4. レンダリングされた画像を保存するためのNumPy配列を事前に確保 (メモリ効率の最重要ポイント)
    # 形状は (N, H, W, 2) になります（画像(1ch) + 壁(1ch)）
    observations = np.empty((num_locations, H, W, 2), dtype=np.float32) 
    
    print(f"💾 出力配列をメモリに事前確保中... サイズ: {observations.nbytes / 1024**3:.2f} GB")


    # 5. レンダリングをバッチごとに行い、事前に確保した配列に直接書き込む
    for i in tqdm(range(0, num_locations, args.render_batch_size), desc="Rendering Batches"):
        sl = slice(i, min(i + args.render_batch_size, num_locations))
        traj_slice = locations[sl]
        
        # 1. 画像のレンダリング
        # images_tensor の形状: (batch_size, H, W)
        images_tensor = ds.render_location(torch.from_numpy(traj_slice))

        # 2. Wall情報の付与
        batch_size = images_tensor.shape[0] # <--- 現在のバッチサイズを取得 (エラー修正ポイント)
        
        # walls_numpyをPyTorchテンソルに戻し、バッチ次元(次元0)を追加して (1, H, W, 1) にする
        walls_batch_dim = torch.from_numpy(walls_numpy).unsqueeze(0) 

        # バッチ次元で現在のバッチサイズ分リピートし、形状を (batch_size, H, W, 1) にする
        repeated_walls = walls_batch_dim.repeat(batch_size, 1, 1, 1)

        # 3. 画像と壁を結合
        # images_tensor の形状を (batch_size, H, W, 1) にする
        images_tensor = images_tensor.unsqueeze(-1)
        
        # dim=-1 (次元3, チャンネル次元) で結合 -> (batch_size, H, W, 2)
        combined_images = torch.cat([images_tensor, repeated_walls], dim=-1)

        # 4. NumPy配列に変換し、事前に確保したobservations配列の対応するスライスに書き込む
        observations[sl] = combined_images.numpy()

    # 6. 最終的な辞書にレンダリング結果を追加
    final_data["observations"] = observations
    final_data["walls"] = walls_numpy # Wall情報も別途保存

    # 7. すべてのデータを一度に保存
    print(f"🎉 処理完了。データを {args.output_path} に保存します...")
    np.savez(args.output_path, **final_data)


if __name__ == "__main__":
    main()
