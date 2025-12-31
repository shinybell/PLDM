"""
MiniGrid環境で学習済みPLDMモデルを評価するスクリプト

使用例:
    python pldm_envs/minigrid/evaluate_model.py \
        --checkpoint checkpoints/minigrid_level1_test/latest.ckpt \
        --env_name MiniGrid-LongHorizon-Level1-v0 \
        --n_episodes 100 \
        --obs_size 72
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import gymnasium as gym

from pldm.models.hjepa import HJEPA
from pldm.train import TrainConfig
from minigrid.wrappers import RGBImgObsWrapper
from pldm_envs.minigrid.wrappers import ResizeObservationWrapper


def load_model(checkpoint_path: str, config_path: str = None):
    """
    チェックポイントからモデルをロード

    Args:
        checkpoint_path: チェックポイントファイルのパス
        config_path: 設定ファイルのパス（オプション）

    Returns:
        model: ロードされたモデル
        config: トレーニング設定
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 設定をロード（チェックポイントまたは設定ファイルから）
    if config_path:
        from omegaconf import OmegaConf
        config_dict = OmegaConf.load(config_path)
    elif 'config' in checkpoint:
        # チェックポイントに設定が含まれている場合
        config = checkpoint['config']
        # モデルを作成
        model = HJEPA(config.hjepa)

        # 重みをロード
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
    # TrainConfig.__post_init__を回避するため、直接hjepaにアクセス
    from omegaconf import OmegaConf

    # SimpleNamespaceとして扱う（TrainConfigの初期化をスキップ）
    class SimpleConfig:
        def __init__(self, hjepa_config):
            self.hjepa = hjepa_config

    config = SimpleConfig(config_dict.hjepa)

    # モデルを作成
    model = HJEPA(config.hjepa)

    # 重みをロード
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # チェックポイント自体がstate_dictの場合
        model.load_state_dict(checkpoint)

    model.eval()
    return model, config


def preprocess_observation(obs: np.ndarray, normalize: bool = True):
    """
    観測を前処理してモデル入力形式に変換

    Args:
        obs: [H, W, C] numpy array (uint8)
        normalize: [0, 1]に正規化するか

    Returns:
        preprocessed: [1, C, H, W] torch tensor
    """
    # HWC -> CHW
    obs = np.transpose(obs, (2, 0, 1))

    # Tensorに変換
    obs = torch.from_numpy(obs).float()

    # 正規化
    if normalize:
        obs = obs / 255.0

    # バッチ次元を追加
    obs = obs.unsqueeze(0)

    return obs


def select_action_greedy(model, state, hidden_state=None):
    """
    モデルを使って貪欲にアクションを選択

    Args:
        model: PLDMモデル
        state: 現在の状態 [1, C, H, W]
        hidden_state: RNNの隠れ状態

    Returns:
        action: 選択されたアクション (int)
        hidden_state: 更新された隠れ状態
    """
    with torch.no_grad():
        # エンコード
        z = model.level1.backbone(state)  # [1, feature_dim]

        if hidden_state is None:
            # 初期隠れ状態
            hidden_state = torch.zeros(
                1, 1, model.level1.predictor.rnn.hidden_size
            )

        # 全アクションの価値を予測（簡易版）
        # ここでは各アクションでの次状態の予測を行い、
        # 最も変化が大きいアクションを選択（探索的）
        num_actions = 7
        action_scores = []

        for action in range(num_actions):
            # アクションをone-hotに
            action_onehot = torch.zeros(1, num_actions)
            action_onehot[0, action] = 1.0

            # 次状態を予測
            z_next, _ = model.level1.predictor(
                z, action_onehot, hidden_state
            )

            # 変化の大きさをスコアとする
            score = torch.norm(z_next - z).item()
            action_scores.append(score)

        # ランダム性を加える（epsilon-greedy）
        if np.random.random() < 0.1:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(action_scores)

    return action, hidden_state


def evaluate_episodes(
    model,
    env_name: str,
    n_episodes: int = 100,
    obs_size: int = 72,
    max_steps: int = 1000,
    render: bool = False,
    device: str = 'cpu',
):
    """
    複数エピソードでモデルを評価

    Args:
        model: PLDMモデル
        env_name: 環境名
        n_episodes: 評価エピソード数
        obs_size: 観測画像サイズ
        max_steps: 最大ステップ数
        render: 描画するか
        device: デバイス

    Returns:
        results: 評価結果の辞書
    """
    model = model.to(device)

    # 環境を作成
    env = gym.make(env_name, render_mode='rgb_array' if render else None)
    env = RGBImgObsWrapper(env)
    if obs_size != 64:
        env = ResizeObservationWrapper(env, size=obs_size)

    results = {
        'success_rate': [],
        'episode_lengths': [],
        'episode_rewards': [],
    }

    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        obs, info = env.reset()
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
            action, hidden_state = select_action_greedy(
                model, state, hidden_state
            )

            # 環境を実行
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        # 結果を記録
        success = done and total_reward > 0
        results['success_rate'].append(1.0 if success else 0.0)
        results['episode_lengths'].append(steps)
        results['episode_rewards'].append(total_reward)

    env.close()

    # 統計を計算
    results['mean_success_rate'] = np.mean(results['success_rate'])
    results['std_success_rate'] = np.std(results['success_rate'])
    results['mean_episode_length'] = np.mean(results['episode_lengths'])
    results['std_episode_length'] = np.std(results['episode_lengths'])
    results['mean_episode_reward'] = np.mean(results['episode_rewards'])
    results['std_episode_reward'] = np.std(results['episode_rewards'])

    return results


def print_results(results):
    """評価結果を表示"""
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Success Rate: {results['mean_success_rate']*100:.2f}% ± {results['std_success_rate']*100:.2f}%")
    print(f"Episode Length: {results['mean_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
    print(f"Episode Reward: {results['mean_episode_reward']:.3f} ± {results['std_episode_reward']:.3f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate PLDM model on MiniGrid")
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
        "--n_episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate"
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
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (JSON)"
    )

    args = parser.parse_args()

    # チェックポイントの存在確認
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}")
    model, config = load_model(args.checkpoint, args.config)

    print(f"\nEvaluating on {args.env_name}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Observation size: {args.obs_size}x{args.obs_size}")

    # 評価を実行
    results = evaluate_episodes(
        model=model,
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        obs_size=args.obs_size,
        max_steps=args.max_steps,
        device=args.device,
    )

    # 結果を表示
    print_results(results)

    # 結果を保存
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ndarrayをリストに変換
        save_results = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in results.items()
        }

        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
