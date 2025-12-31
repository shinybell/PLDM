"""
Atari環境のデータセット設定とサンプル定義

参考:
- Gymnasium Atari: https://gymnasium.farama.org/environments/atari/
- ALE-py: https://pypi.org/project/ale-py/
"""

from typing import NamedTuple, Optional
import torch
from dataclasses import dataclass
from omegaconf import MISSING


class AtariSample(NamedTuple):
    """
    Atariデータセットの1サンプルを表すNamedTuple

    Fields:
        states: 観測画像 [T, C, H, W] (RGB or Grayscale)
        actions: アクション系列 [T-1, 1] (離散アクション)
        rewards: 報酬系列 [T-1] (オプション)
        dones: 終端フラグ [T-1] (オプション)
    """
    states: torch.Tensor  # [T, C, H, W]
    actions: torch.Tensor  # [T-1, 1]
    rewards: Optional[torch.Tensor] = None  # [T-1]
    dones: Optional[torch.Tensor] = None  # [T-1]


@dataclass
class AtariDatasetConfig:
    """
    Atariデータセットの設定

    主要パラメータ:
        env_name: Atari環境名 (例: "MsPacman-v5", "Pong-v5")
        batch_size: バッチサイズ
        sample_length: サンプルのタイムステップ数
        online_mode: オンライン生成モードか保存済みデータモードか

    データソース:
        data_path: オフラインデータのパス (.npz, .pkl)

    画像設定:
        img_size: リサイズ後の画像サイズ
        grayscale: グレースケール変換するか
        frame_stack: フレームスタック数 (DQNスタイル)
        frame_skip: フレームスキップ数

    Gymnasium設定:
        obs_type: "rgb" or "grayscale"
        repeat_action_probability: アクション繰り返し確率
        full_action_space: 全18アクションを使うか
    """
    # 環境設定
    env_name: str = "ALE/Pacman-v5"
    batch_size: int = 32
    sample_length: int = 17  # タイムステップ数
    train: bool = True

    # データソース
    online_mode: bool = False  # True: オンライン生成, False: オフラインデータ
    data_path: Optional[str] = None  # オフラインデータのパス
    val_path: Optional[str] = None  # 検証データのパス

    # 画像設定
    img_size: int = 64  # リサイズ後のサイズ
    grayscale: bool = False  # True: グレースケール, False: RGB
    frame_stack: int = 1  # フレームスタック数 (1: スタックなし)
    frame_skip: int = 4  # フレームスキップ

    # Gymnasium設定
    obs_type: str = "rgb"  # "rgb" or "grayscale"
    repeat_action_probability: float = 0.0  # Sticky actions
    full_action_space: bool = False  # 全18アクションを使うか
    render_mode: Optional[str] = None  # "human", "rgb_array", None

    # オンライン生成設定
    n_episodes: int = 1000  # オンライン生成時のエピソード数
    max_episode_steps: int = 1000  # 最大ステップ数
    policy_type: str = "random"  # "random", "trained", "human"
    policy_path: Optional[str] = None  # 訓練済みポリシーのパス

    # データセット管理
    num_workers: int = 0
    crop_length: Optional[int] = None  # データセット長の制限
    quick_debug: bool = False
    seed: int = 0

    # 前処理
    normalize_images: bool = True  # 0-1に正規化
    include_rewards: bool = False  # 報酬を含めるか
    include_dones: bool = False  # 終端フラグを含めるか

    def __post_init__(self):
        """設定の整合性チェック"""
        if self.grayscale:
            self.obs_type = "grayscale"

        # NOTE: data_pathの検証はデータセット作成時に行う
        # TrainConfigの初期化時にはチェックしない
        # if not self.online_mode and self.data_path is None:
        #     raise ValueError(
        #         "offline mode requires data_path to be specified"
        #     )
