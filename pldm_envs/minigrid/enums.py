"""
MiniGrid環境のデータセット設定

PLDM学習用のMiniGrid設定を定義します。
"""

# Import the config from the data module to avoid duplication
from pldm_envs.minigrid.data.minigrid_dataset import MiniGridDatasetConfig

__all__ = [
    "MiniGridDatasetConfig",
]
