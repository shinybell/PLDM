"""
MiniGrid環境で学習済みPLDMモデルのエージェントの動きを可視化

使用例:
    # 単一エピソードをGIFで保存
    python pldm_envs/minigrid/visualize_agent.py \
        --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
        --env_name MiniGrid-LongHorizon-Level1-v0 \
        --output visualizations/agent_episode.gif \
        --obs_size 72

    # 複数エピソードをビデオで保存
    python pldm_envs/minigrid/visualize_agent.py \
        --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
        --env_name MiniGrid-LongHorizon-Level1-v0 \
        --output visualizations/agent_episodes.mp4 \
        --n_episodes 5 \
        --obs_size 72
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import gymnasium as gym
from PIL import Image
import imageio

from pldm.models.hjepa import HJEPA
from pldm.train import TrainConfig
from minigrid.wrappers import RGBImgObsWrapper
from pldm_envs.minigrid.wrappers import ResizeObservationWrapper


def load_model(checkpoint_path: str, config_path: str = None):
    """チェックポイントからモデルをロード"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if config_path:
        from omegaconf import OmegaConf
        from pldm.models.hjepa import HJEPAConfig
        from pldm.models.encoders.enums import BackboneConfig
        from pldm.models.enums import PredictorConfig

        # データクラスの構造を使ってデフォルト値を含む設定を作成
        hjepa_structured = OmegaConf.structured(HJEPAConfig)
        yaml_config = OmegaConf.load(config_path)
        # YAMLの設定とマージ（YAMLに無いフィールドはデフォルト値が使われる）
        config_dict = OmegaConf.merge(hjepa_structured, yaml_config)
    elif 'config' in checkpoint:
        # チェックポイントに設定が含まれている場合
        config = checkpoint['config']

        # 入力次元を設定から取得
        if hasattr(config, 'data') and hasattr(config.data, 'minigrid_config'):
            img_size = config.data.minigrid_config.img_size
        else:
            img_size = 72  # デフォルト値
        channels = 3  # RGB
        input_dim = (channels, img_size, img_size)

        model = HJEPA(config.hjepa, input_dim=input_dim)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model, config
    else:
        # チェックポイントと同じディレクトリにある設定ファイルを探す
        checkpoint_dir = Path(checkpoint_path).parent.parent / "configs" / "minigrid"
        possible_configs = list(checkpoint_dir.glob("*.yaml"))
        if possible_configs:
            print(f"Config not found in checkpoint. Using {possible_configs[0]}")
            from omegaconf import OmegaConf
            config_dict = OmegaConf.load(possible_configs[0])
        else:
            raise ValueError(
                "Config not found in checkpoint and no config_path provided. "
                "Please specify --config path/to/config.yaml"
            )

    # OmegaConfの辞書から直接モデル設定を取得
    # TrainConfig.__post_init__を回避
    class SimpleConfig:
        def __init__(self, hjepa_config):
            self.hjepa = hjepa_config

    config = SimpleConfig(config_dict.hjepa)

    # 入力次元を設定から取得
    img_size = config_dict.data.minigrid_config.img_size  # 72
    channels = 3  # RGB
    input_dim = (channels, img_size, img_size)

    model = HJEPA(config.hjepa, input_dim=input_dim)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, config


def preprocess_observation(obs: np.ndarray, normalize: bool = True):
    """観測を前処理"""
    obs = np.transpose(obs, (2, 0, 1))
    obs = torch.from_numpy(obs).float()
    if normalize:
        obs = obs / 255.0
    obs = obs.unsqueeze(0)
    return obs


def select_action(model, state, hidden_state=None, mode='greedy', epsilon=0.1):
    """
    アクションを選択

    Args:
        model: PLDMモデル
        state: 状態 [1, C, H, W]
        hidden_state: 隠れ状態
        mode: 'greedy', 'random', or 'epsilon-greedy'
        epsilon: epsilon-greedyの場合のepsilon値

    Returns:
        action: 選択されたアクション
        hidden_state: 更新された隠れ状態
    """
    num_actions = 7

    if mode == 'random':
        return np.random.randint(num_actions), hidden_state

    with torch.no_grad():
        z = model.level1.backbone(state)

        if hidden_state is None:
            hidden_state = torch.zeros(
                1, 1, model.level1.predictor.rnn.hidden_size
            )

        # 各アクションのスコアを計算
        action_scores = []
        for action in range(num_actions):
            action_onehot = torch.zeros(1, num_actions)
            action_onehot[0, action] = 1.0
            z_next, _ = model.level1.predictor(z, action_onehot, hidden_state)
            score = torch.norm(z_next - z).item()
            action_scores.append(score)

        if mode == 'epsilon-greedy' and np.random.random() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(action_scores)

    return action, hidden_state


def run_episode(
    model,
    env,
    obs_size: int = 72,
    max_steps: int = 1000,
    device: str = 'cpu',
    action_mode: str = 'epsilon-greedy',
    epsilon: float = 0.1,
):
    """
    1エピソードを実行してフレームを記録

    Returns:
        frames: フレームのリスト
        info: エピソード情報
    """
    model = model.to(device)

    obs, info = env.reset()
    frames = [env.render()]
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    hidden_state = None

    while not (done or truncated) and steps < max_steps:
        # 観測を前処理
        state = preprocess_observation(obs, normalize=True)
        state = state.to(device)

        # アクションを選択
        action, hidden_state = select_action(
            model, state, hidden_state, mode=action_mode, epsilon=epsilon
        )

        # 環境を実行
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # フレームを記録
        frames.append(env.render())

    episode_info = {
        'success': done and total_reward > 0,
        'steps': steps,
        'reward': total_reward,
    }

    return frames, episode_info


def save_as_gif(frames, output_path: str, fps: int = 10):
    """フレームをGIFとして保存"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # PIL Imageに変換
    images = [Image.fromarray(frame) for frame in frames]

    # GIFとして保存
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"Saved GIF to {output_path}")


def save_as_video(frames, output_path: str, fps: int = 10):
    """フレームをビデオとして保存"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # imageioを使ってビデオを保存
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved video to {output_path}")


def save_as_images(frames, output_dir: str, prefix: str = "frame"):
    """フレームを個別画像として保存"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        image = Image.fromarray(frame)
        image.save(output_dir / f"{prefix}_{i:04d}.png")

    print(f"Saved {len(frames)} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize PLDM agent on MiniGrid")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional)"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="MiniGrid-LongHorizon-Level1-v0",
        help="MiniGrid environment name"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (.gif, .mp4, or directory for images)"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=1,
        help="Number of episodes to visualize"
    )
    parser.add_argument(
        "--obs_size",
        type=int,
        default=72,
        help="Observation image size"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for video/GIF"
    )
    parser.add_argument(
        "--action_mode",
        type=str,
        default="epsilon-greedy",
        choices=["greedy", "random", "epsilon-greedy"],
        help="Action selection mode"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon for epsilon-greedy"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )

    args = parser.parse_args()

    # チェックポイントをロード
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}")
    model, config = load_model(args.checkpoint, args.config)

    # 環境を作成
    env = gym.make(args.env_name, render_mode='rgb_array')
    env = RGBImgObsWrapper(env)
    if args.obs_size != 64:
        env = ResizeObservationWrapper(env, size=args.obs_size)

    print(f"\nRunning {args.n_episodes} episode(s) on {args.env_name}")

    all_frames = []
    episode_infos = []

    for episode in range(args.n_episodes):
        print(f"\nEpisode {episode + 1}/{args.n_episodes}")

        frames, info = run_episode(
            model=model,
            env=env,
            obs_size=args.obs_size,
            max_steps=args.max_steps,
            device=args.device,
            action_mode=args.action_mode,
            epsilon=args.epsilon,
        )

        all_frames.extend(frames)
        episode_infos.append(info)

        print(f"  Success: {info['success']}")
        print(f"  Steps: {info['steps']}")
        print(f"  Reward: {info['reward']:.3f}")

        # エピソード間に区切りフレームを追加（複数エピソードの場合）
        if args.n_episodes > 1 and episode < args.n_episodes - 1:
            # 黒いフレームを数フレーム追加
            black_frame = np.zeros_like(frames[0])
            all_frames.extend([black_frame] * 5)

    env.close()

    # 出力形式を決定
    output_path = Path(args.output)
    if output_path.suffix == '.gif':
        save_as_gif(all_frames, output_path, fps=args.fps)
    elif output_path.suffix in ['.mp4', '.avi']:
        save_as_video(all_frames, output_path, fps=args.fps)
    else:
        # ディレクトリとして扱い、個別画像を保存
        save_as_images(all_frames, output_path)

    # サマリーを表示
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    success_count = sum(1 for info in episode_infos if info['success'])
    print(f"Success Rate: {success_count}/{args.n_episodes} ({success_count/args.n_episodes*100:.1f}%)")
    print(f"Average Steps: {np.mean([info['steps'] for info in episode_infos]):.1f}")
    print(f"Average Reward: {np.mean([info['reward'] for info in episode_infos]):.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
