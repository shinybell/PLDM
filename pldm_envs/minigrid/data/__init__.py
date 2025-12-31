"""MiniGrid data module"""

import torch
from pldm_envs.minigrid.data.minigrid_dataset import (
    MiniGridDataset,
    MiniGridDatasetConfig,
    MiniGridSample,
)


def minigrid_collate_fn(batch):
    """
    MiniGridSampleのバッチをまとめるcollate関数

    Args:
        batch: List[MiniGridSample]

    Returns:
        MiniGridSample: バッチ化されたサンプル
    """
    states = torch.stack([sample.states for sample in batch])
    actions = torch.stack([sample.actions for sample in batch])

    # rewardsとdonesは全てNoneまたは全て有効値
    if batch[0].rewards is not None:
        rewards = torch.stack([sample.rewards for sample in batch])
    else:
        rewards = None

    if batch[0].dones is not None:
        dones = torch.stack([sample.dones for sample in batch])
    else:
        dones = None

    return MiniGridSample(
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )


__all__ = [
    "MiniGridDataset",
    "MiniGridDatasetConfig",
    "MiniGridSample",
    "minigrid_collate_fn",
]
