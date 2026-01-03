"""
DiverseMaze環境から直接観測画像をレンダリングして可視化するスクリプト
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

try:
    import gym
    import d4rl
    import d4rl.pointmaze
    from pldm_envs.diverse_maze.maze_draw import render_umaze, create_drawer
    from pldm_envs.diverse_maze.transforms import select_transforms
    from PIL import Image

    # 環境を作成
    env_name = "maze2d-umaze-v1"  # 基本的なU字型迷路
    print(f"Creating environment: {env_name}")

    env = gym.make(env_name)
    env.name = env_name

    # 環境をリセット
    obs = env.reset()
    print(f"Environment observation shape: {obs.shape}")
    print(f"Environment observation: {obs}")

    # Drawerを作成
    drawer = create_drawer(env, env_name)
    print(f"Drawer created for {env_name}")

    # 変換を取得
    transforms = select_transforms(env_name)
    print(f"Transforms: {transforms}")

    # 複数の状態をサンプリングして画像化
    n_samples = 5
    images = []
    states = []

    for i in range(n_samples):
        # ランダムな状態を生成（x, y, vx, vy）
        # 迷路内の有効な位置をサンプリング
        if i == 0:
            # 初期状態
            state = obs
        else:
            # ランダムにステップを実行
            for _ in range(np.random.randint(5, 20)):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
            state = obs

        states.append(state)

        # 状態から画像をレンダリング
        image_pil = Image.fromarray(np.uint8(drawer.render_state(state)))
        image_transformed = transforms(image_pil)
        image_tensor = torch.from_numpy(np.array(image_transformed)).permute(2, 0, 1)

        images.append(image_tensor)
        print(f"Sample {i}: state shape {state.shape}, image shape {image_tensor.shape}, "
              f"range [{image_tensor.min()}, {image_tensor.max()}]")

    # 可視化
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))

    for i in range(n_samples):
        # 元のレンダリング（変換前）
        raw_image = drawer.render_state(states[i])
        axes[0, i].imshow(raw_image.astype(np.uint8))
        axes[0, i].set_title(f"Raw {i}\n{raw_image.shape}")
        axes[0, i].axis('off')

        # 変換後（64x64）
        img = images[i].permute(1, 2, 0).numpy().astype(np.uint8)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Transformed {i}\n{img.shape}")
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Raw Render", fontsize=12)
    axes[1, 0].set_ylabel("64x64 Resized", fontsize=12)

    plt.tight_layout()
    plt.savefig("diverse_maze_env_samples.png", dpi=150, bbox_inches='tight')
    print(f"\n可視化結果を 'diverse_maze_env_samples.png' に保存しました")
    plt.close()

    # 詳細画像
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(drawer.render_state(states[0]).astype(np.uint8))
    plt.title(f"Raw Render\n{drawer.render_state(states[0]).shape}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    img = images[0].permute(1, 2, 0).numpy().astype(np.uint8)
    plt.imshow(img)
    plt.title(f"64x64 Transformed\n{img.shape}")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("diverse_maze_comparison.png", dpi=150, bbox_inches='tight')
    print("比較画像を 'diverse_maze_comparison.png' に保存しました")

except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    print("d4rlとpldm_envsをインストールしてください")
except Exception as e:
    print(f"エラーが発生しました: {e}")
    import traceback
    traceback.print_exc()
