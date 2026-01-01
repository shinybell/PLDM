"""MiniGrid Planning Configuration and Report Types"""
from dataclasses import dataclass
from typing import Optional, NamedTuple
import torch
from pldm.planning.enums import MPCConfig


@dataclass
class MiniGridMPCConfig(MPCConfig):
    """
    MiniGrid MPC評価の設定

    Attributes:
        discount: 割引率
        level: 難易度レベル ('level1', 'level2', 'level3')
        seed: ランダムシード
        error_threshold: 目標到達判定の閾値（座標の距離）
        max_episode_steps: エピソードの最大ステップ数
    """
    discount: float = 1.0
    level: str = "level1"
    seed: Optional[int] = 42
    error_threshold: float = 1.0
    max_episode_steps: int = 1000


class MPCReport(NamedTuple):
    """
    MiniGrid MPC評価の結果レポート

    Attributes:
        error_mean: 最終誤差の平均
        errors: 各エピソードの最終誤差
        terminations: 各エピソードの終了ステップ
        planning_time: プランニングにかかった時間（秒）
        success_rate: ゴール到達率
        avg_steps: 平均ステップ数
    """
    error_mean: torch.Tensor
    errors: torch.Tensor
    terminations: list
    planning_time: int
    success_rate: float
    avg_steps: float

    def build_log_dict(self, prefix: str = ""):
        """
        ログ用の辞書を作成

        Args:
            prefix: ログキーのプレフィックス

        Returns:
            ログ辞書
        """
        return {
            f"{prefix}planning_error_mean": self.error_mean.item(),
            f"{prefix}planning_error_mean_rmse": self.error_mean.pow(0.5).item(),
            f"{prefix}success_rate": self.success_rate,
            f"{prefix}avg_steps": self.avg_steps,
            f"{prefix}avg_termination_step": sum(self.terminations)
            / len(self.terminations),
        }
