"""
MiniGrid環境のデータセット設定とサンプル定義

参考:
- MiniGrid Documentation: https://minigrid.farama.org/
- Gymnasium: https://gymnasium.farama.org/
"""

from typing import NamedTuple, Optional
import torch
from dataclasses import dataclass


class MiniGridSample(NamedTuple):
    """
    MiniGridデータセットの1サンプルを表すNamedTuple

    Fields:
        states: 観測画像 [T, C, H, W] (RGB)
        actions: アクション系列 [T-1, 1] (離散アクション)
        rewards: 報酬系列 [T-1] (オプション)
        dones: 終端フラグ [T-1] (オプション)
    """
    states: torch.Tensor  # [T, C, H, W]
    actions: torch.Tensor  # [T-1, 1]
    rewards: Optional[torch.Tensor] = None  # [T-1]
    dones: Optional[torch.Tensor] = None  # [T-1]


@dataclass
class MiniGridDatasetConfig:
    """
    MiniGridデータセットの設定

    主要パラメータ:
        env_name: MiniGrid環境名 (例: "MiniGrid-Empty-8x8-v0")
        batch_size: バッチサイズ
        sample_length: サンプルのタイムステップ数
        online_mode: オンライン生成モードか保存済みデータモードか

    データソース:
        data_path: オフラインデータのパス (.npz, .pkl)

    画像設定:
        img_size: リサイズ後の画像サイズ
        tile_size: MiniGridのタイルサイズ (デフォルト8)
        agent_view_size: エージェントの視界サイズ (デフォルトNone=全体)

    環境設定:
        max_steps: エピソードの最大ステップ数
        render_mode: レンダリングモード
    """
    # 環境設定
    env_name: str = "MiniGrid-Empty-8x8-v0"
    batch_size: int = 32
    sample_length: int = 17  # タイムステップ数
    train: bool = True

    # データソース
    online_mode: bool = False  # True: オンライン生成, False: オフラインデータ
    data_path: Optional[str] = None  # オフラインデータのパス
    val_path: Optional[str] = None  # 検証データのパス

    # 画像設定
    img_size: int = 64  # リサイズ後のサイズ
    tile_size: int = 8  # MiniGridのタイルサイズ
    agent_view_size: Optional[int] = None  # None=全体を観測, 7=部分観測(7x7)

    # MiniGrid設定
    max_steps: Optional[int] = None  # None=デフォルト値を使用
    render_mode: Optional[str] = None  # "human", "rgb_array", None
    highlight: bool = False  # エージェントの視界をハイライトするか

    # オンライン生成設定
    n_episodes: int = 1000  # オンライン生成時のエピソード数
    policy_type: str = "random"  # "random", "shortest_path", "trained"
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
        if not self.online_mode and self.data_path is None:
            raise ValueError(
                "offline mode requires data_path to be specified"
            )
