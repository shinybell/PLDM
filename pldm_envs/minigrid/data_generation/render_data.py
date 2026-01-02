"""
MiniGridのプロプリオセプティブデータ（.p）から画像をレンダリングするスクリプト

Diverse Mazeのrender_data.pyを参考にMiniGrid用に実装。

使用方法:
    python pldm_envs/minigrid/data_generation/render_data.py \\
        --data_path pldm_envs/minigrid/data/sample_10episodes
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import torch
import argparse
import gymnasium as gym
import minigrid
from PIL import Image

# MiniGridの登録
gym.register_envs(minigrid)
import pldm_envs.minigrid


def save_image(array, path):
    """numpy配列を画像として保存"""
    image = Image.fromarray(np.uint8(array))
    image.save(path, format="png")


def render_minigrid_state(env, agent_pos, agent_dir):
    """
    MiniGrid環境でエージェントの位置と向きから画像をレンダリング

    Args:
        env: MiniGrid環境
        agent_pos: エージェント位置 [x, y]
        agent_dir: エージェント向き (0-3)

    Returns:
        RGB画像 [H, W, 3]
    """
    # エージェントの位置と向きを設定
    env.unwrapped.agent_pos = tuple(agent_pos.astype(int))
    env.unwrapped.agent_dir = int(agent_dir) if not isinstance(agent_dir, (int, np.integer)) else agent_dir

    # 画像を取得
    image = env.unwrapped.get_frame(
        highlight=False,
        tile_size=env.unwrapped.tile_size
    )

    return image


def main():
    parser = argparse.ArgumentParser(description="MiniGridのプロプリオデータから画像をレンダリング")
    parser.add_argument("--data_path", type=str, required=True, help="データパス")
    parser.add_argument("--save_replace", action="store_true", help="既存画像を上書き")
    parser.add_argument("--quick_debug", action="store_true", help="デバッグモード（少量データのみ処理）")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_image_path = data_path / "images"
    output_image_path.mkdir(parents=True, exist_ok=True)

    propio_path = data_path / "data.p"
    config_path = data_path / "metadata.pt"

    # 設定とデータを読み込み
    if not config_path.exists():
        print(f"Error: metadata.pt not found at {config_path}")
        return

    if not propio_path.exists():
        print(f"Error: data.p not found at {propio_path}")
        return

    config = torch.load(config_path)
    all_splits = torch.load(propio_path)

    print(f"Total episodes: {len(all_splits)}")
    print(f"Config: {config}")

    # 環境を作成
    env_name = config.get("env_name", "MiniGrid-Empty-8x8-v0")
    tile_size = config.get("tile_size", 8)

    env = gym.make(
        env_name,
        tile_size=tile_size,
        highlight=False,
        render_mode=None,
    )

    print(f"Environment: {env_name}")
    print(f"Tile size: {tile_size}")

    # 各エピソードの各タイムステップについて画像を生成
    for split_idx, split in enumerate(tqdm(all_splits, desc="Rendering episodes")):
        positions = split["positions"]  # [T, 2] - エージェント位置

        # MiniGridではagent_dirも必要なので、アクションから推定するか、データに含める必要がある
        # 簡易実装: agent_dir=0（右向き）で固定
        # より正確にするには、generate_data.pyでagent_dirも保存する必要がある

        for img_idx, pos in enumerate(tqdm(positions, desc=f"Episode {split_idx}", leave=False)):
            image_path = output_image_path / f"{split_idx}_{img_idx}.png"

            if os.path.exists(image_path) and not args.save_replace:
                continue

            # 画像をレンダリング
            # 注意: posは[x, y]、agent_dirは0で固定
            image = render_minigrid_state(env, pos, agent_dir=0)
            save_image(image, image_path)

            if args.quick_debug and img_idx > 10:
                break

        if args.quick_debug and split_idx > 2:
            break

    env.close()
    print(f"\nRendering completed! Images saved to {output_image_path}")


if __name__ == "__main__":
    main()
